/*
 * transformer.c — Scaled dot-product attention + Transformer block (float64)
 *
 * Batch 13 feature set:
 *   - attention_scaled_dot_product
 *   - transformer_block_forward (single-head encoder block)
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double gelu_approx(double x)
{
    const double c = sqrt(2.0 / M_PI);
    const double x3 = x * x * x;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x3)));
}

static int matmul_add_bias(const double *a, const double *w, const double *b,
                           int64_t m, int64_t k, int64_t n,
                           double *out)
{
    if (!a || !w || !out) return -1;
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = b ? b[j] : 0.0;
            for (int64_t t = 0; t < k; ++t) {
                acc += a[i * k + t] * w[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return 0;
}

static void layernorm_last_dim(const double *in,
                               int64_t rows,
                               int64_t cols,
                               double eps,
                               double *out)
{
    for (int64_t r = 0; r < rows; ++r) {
        const double *src = in + r * cols;
        double *dst = out + r * cols;

        double mean = 0.0;
        for (int64_t c = 0; c < cols; ++c) mean += src[c];
        mean /= (double)cols;

        double var = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            const double d = src[c] - mean;
            var += d * d;
        }
        var /= (double)cols;

        const double inv_std = 1.0 / sqrt(var + eps);
        for (int64_t c = 0; c < cols; ++c) {
            dst[c] = (src[c] - mean) * inv_std;
        }
    }
}

int attention_scaled_dot_product(const double *q,
                                 const double *k,
                                 const double *v,
                                 const double *mask,
                                 int64_t batch,
                                 int64_t heads,
                                 int64_t seq_q,
                                 int64_t seq_kv,
                                 int64_t d_k,
                                 int64_t d_v,
                                 double *out,
                                 double *attn_weights)
{
    if (!q || !k || !v || !out) return -1;
    if (batch <= 0 || heads <= 0 || seq_q <= 0 || seq_kv <= 0 || d_k <= 0 || d_v <= 0) return -1;

    double *row_scores = (double *)malloc((size_t)seq_kv * sizeof(double));
    if (!row_scores) return -1;

    const double scale = 1.0 / sqrt((double)d_k);
    const int64_t head_q_stride = seq_q * d_k;
    const int64_t head_k_stride = seq_kv * d_k;
    const int64_t head_v_stride = seq_kv * d_v;
    const int64_t head_out_stride = seq_q * d_v;
    const int64_t head_attn_stride = seq_q * seq_kv;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t base_q = (b * heads + h) * head_q_stride;
            const int64_t base_k = (b * heads + h) * head_k_stride;
            const int64_t base_v = (b * heads + h) * head_v_stride;
            const int64_t base_o = (b * heads + h) * head_out_stride;
            const int64_t base_a = (b * heads + h) * head_attn_stride;

            for (int64_t i = 0; i < seq_q; ++i) {
                double row_max = -1e300;

                for (int64_t j = 0; j < seq_kv; ++j) {
                    double score = 0.0;
                    const double *q_i = q + base_q + i * d_k;
                    const double *k_j = k + base_k + j * d_k;
                    for (int64_t t = 0; t < d_k; ++t) {
                        score += q_i[t] * k_j[t];
                    }
                    score *= scale;
                    if (mask) score += mask[base_a + i * seq_kv + j];

                    row_scores[j] = score;
                    if (attn_weights) attn_weights[base_a + i * seq_kv + j] = score;
                    if (score > row_max) row_max = score;
                }

                double denom = 0.0;
                for (int64_t j = 0; j < seq_kv; ++j) {
                    double e = exp(row_scores[j] - row_max);
                    if (attn_weights) {
                        attn_weights[base_a + i * seq_kv + j] = e;
                    }
                    denom += e;
                }
                if (denom <= 0.0) denom = 1.0;

                for (int64_t j = 0; j < seq_kv; ++j) {
                    double p = 0.0;
                    if (attn_weights) {
                        p = attn_weights[base_a + i * seq_kv + j] / denom;
                        attn_weights[base_a + i * seq_kv + j] = p;
                    } else {
                        p = exp(row_scores[j] - row_max) / denom;
                    }

                    const double *v_j = v + base_v + j * d_v;
                    double *o_i = out + base_o + i * d_v;
                    if (j == 0) {
                        for (int64_t dv = 0; dv < d_v; ++dv) o_i[dv] = 0.0;
                    }
                    for (int64_t dv = 0; dv < d_v; ++dv) {
                        o_i[dv] += p * v_j[dv];
                    }
                }
            }
        }
    }

    free(row_scores);
    return 0;
}

int transformer_block_forward(const double *x,
                              int64_t batch,
                              int64_t seq_len,
                              int64_t d_model,
                              int64_t d_ff,
                              const double *w_q,
                              const double *w_k,
                              const double *w_v,
                              const double *w_o,
                              const double *b_q,
                              const double *b_k,
                              const double *b_v,
                              const double *b_o,
                              const double *w1,
                              const double *b1,
                              const double *w2,
                              const double *b2,
                              double eps,
                              double *out)
{
    if (!x || !w_q || !w_k || !w_v || !w_o || !w1 || !w2 || !out) return -1;
    if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_ff <= 0) return -1;

    const int64_t tokens = batch * seq_len;
    const size_t sz_tok_dm = (size_t)tokens * (size_t)d_model;
    const size_t sz_tok_ff = (size_t)tokens * (size_t)d_ff;

    double *q = (double *)malloc(sz_tok_dm * sizeof(double));
    double *k = (double *)malloc(sz_tok_dm * sizeof(double));
    double *v = (double *)malloc(sz_tok_dm * sizeof(double));
    double *attn = (double *)malloc(sz_tok_dm * sizeof(double));
    double *attn_proj = (double *)malloc(sz_tok_dm * sizeof(double));
    double *res1 = (double *)malloc(sz_tok_dm * sizeof(double));
    double *ln1 = (double *)malloc(sz_tok_dm * sizeof(double));
    double *ff1 = (double *)malloc(sz_tok_ff * sizeof(double));
    double *ff2 = (double *)malloc(sz_tok_dm * sizeof(double));
    double *res2 = (double *)malloc(sz_tok_dm * sizeof(double));

    if (!q || !k || !v || !attn || !attn_proj || !res1 || !ln1 || !ff1 || !ff2 || !res2) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    if (matmul_add_bias(x, w_q, b_q, tokens, d_model, d_model, q) != 0 ||
        matmul_add_bias(x, w_k, b_k, tokens, d_model, d_model, k) != 0 ||
        matmul_add_bias(x, w_v, b_v, tokens, d_model, d_model, v) != 0) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    if (attention_scaled_dot_product(q, k, v,
                                     NULL,
                                     batch, 1,
                                     seq_len, seq_len,
                                     d_model, d_model,
                                     attn, NULL) != 0) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    if (matmul_add_bias(attn, w_o, b_o, tokens, d_model, d_model, attn_proj) != 0) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    for (int64_t i = 0; i < tokens * d_model; ++i) {
        res1[i] = x[i] + attn_proj[i];
    }

    layernorm_last_dim(res1, tokens, d_model, eps, ln1);

    if (matmul_add_bias(ln1, w1, b1, tokens, d_model, d_ff, ff1) != 0) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    for (int64_t i = 0; i < tokens * d_ff; ++i) {
        ff1[i] = gelu_approx(ff1[i]);
    }

    if (matmul_add_bias(ff1, w2, b2, tokens, d_ff, d_model, ff2) != 0) {
        free(q); free(k); free(v); free(attn); free(attn_proj);
        free(res1); free(ln1); free(ff1); free(ff2); free(res2);
        return -1;
    }

    for (int64_t i = 0; i < tokens * d_model; ++i) {
        res2[i] = ln1[i] + ff2[i];
    }

    layernorm_last_dim(res2, tokens, d_model, eps, out);

    free(q); free(k); free(v); free(attn); free(attn_proj);
    free(res1); free(ln1); free(ff1); free(ff2); free(res2);
    return 0;
}
