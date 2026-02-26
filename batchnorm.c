/*
 * batchnorm.c — BatchNorm1d and LayerNorm for Neural-Assembly
 *
 * BatchNorm1d:
 *   Train: normalize using batch mean/var, update running stats via EMA
 *   Eval:  normalize using running mean/var
 *   Learnable parameters: gamma (scale), beta (shift)
 *
 * LayerNorm:
 *   Always uses per-sample stats (no running mean/var)
 *   Learnable parameters: gamma (scale), beta (shift)
 *
 * All arrays are double* for consistency with the framework's float64 path.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* BatchNorm1d state                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int64_t  num_features;
    double   momentum;      /* EMA coefficient (default 0.1)      */
    double   eps;           /* numerical stability (default 1e-5) */
    double  *gamma;         /* scale,  length = num_features      */
    double  *beta;          /* shift,  length = num_features      */
    double  *running_mean;  /* EMA mean, length = num_features    */
    double  *running_var;   /* EMA var,  length = num_features    */
    /* Saved for backward: */
    double  *saved_mean;    /* batch mean from last forward       */
    double  *saved_invstd;  /* 1/sqrt(var+eps) from last forward  */
    double  *x_hat;         /* normalised input (B*C)             */
    int64_t  last_batch;    /* batch size of last forward         */
} BatchNorm1d;

/* ---- create / free ------------------------------------------------ */

BatchNorm1d *batchnorm1d_create(int64_t num_features, double momentum, double eps)
{
    if (num_features <= 0) return NULL;

    BatchNorm1d *bn = calloc(1, sizeof(*bn));
    if (!bn) return NULL;

    bn->num_features = num_features;
    bn->momentum     = momentum;
    bn->eps          = eps;

    bn->gamma        = malloc((size_t)num_features * sizeof(double));
    bn->beta         = calloc((size_t)num_features, sizeof(double));
    bn->running_mean = calloc((size_t)num_features, sizeof(double));
    bn->running_var  = malloc((size_t)num_features * sizeof(double));
    bn->saved_mean   = malloc((size_t)num_features * sizeof(double));
    bn->saved_invstd = malloc((size_t)num_features * sizeof(double));
    bn->x_hat        = NULL;  /* allocated on first forward */
    bn->last_batch   = 0;

    if (!bn->gamma || !bn->beta || !bn->running_mean ||
        !bn->running_var || !bn->saved_mean || !bn->saved_invstd) {
        free(bn->gamma); free(bn->beta);
        free(bn->running_mean); free(bn->running_var);
        free(bn->saved_mean); free(bn->saved_invstd);
        free(bn);
        return NULL;
    }

    /* Initialize gamma=1, running_var=1 */
    for (int64_t i = 0; i < num_features; i++) {
        bn->gamma[i]       = 1.0;
        bn->running_var[i] = 1.0;
    }

    return bn;
}

void batchnorm1d_free(BatchNorm1d *bn)
{
    if (!bn) return;
    free(bn->gamma);
    free(bn->beta);
    free(bn->running_mean);
    free(bn->running_var);
    free(bn->saved_mean);
    free(bn->saved_invstd);
    free(bn->x_hat);
    free(bn);
}

/* ---- forward ------------------------------------------------------ */

/*
 * batchnorm1d_forward
 *
 *   input:   [batch_size × num_features] row-major
 *   output:  [batch_size × num_features] row-major
 *   training: 1 = use batch stats & update running, 0 = use running stats
 *
 * Returns 0 on success, -1 on error.
 */
int batchnorm1d_forward(BatchNorm1d *bn,
                         const double *input, double *output,
                         int64_t batch_size, int training)
{
    if (!bn || !input || !output || batch_size <= 0) return -1;

    int64_t C = bn->num_features;
    double eps = bn->eps;

    /* Allocate x_hat cache if needed */
    int64_t total = batch_size * C;
    if (bn->last_batch != batch_size || !bn->x_hat) {
        free(bn->x_hat);
        bn->x_hat = malloc((size_t)total * sizeof(double));
        if (!bn->x_hat) return -1;
    }
    bn->last_batch = batch_size;

    if (training) {
        /* Compute batch mean and variance per feature */
        for (int64_t c = 0; c < C; c++) {
            double sum  = 0.0;
            for (int64_t b = 0; b < batch_size; b++)
                sum += input[b * C + c];
            double mean = sum / (double)batch_size;

            double var_sum = 0.0;
            for (int64_t b = 0; b < batch_size; b++) {
                double d = input[b * C + c] - mean;
                var_sum += d * d;
            }
            double var = var_sum / (double)batch_size;   /* biased variance */

            bn->saved_mean[c]   = mean;
            double invstd = 1.0 / sqrt(var + eps);
            bn->saved_invstd[c] = invstd;

            /* Normalize, scale, shift */
            for (int64_t b = 0; b < batch_size; b++) {
                double xh = (input[b * C + c] - mean) * invstd;
                bn->x_hat[b * C + c] = xh;
                output[b * C + c] = bn->gamma[c] * xh + bn->beta[c];
            }

            /* Update running stats (EMA) with unbiased variance */
            double m = bn->momentum;
            bn->running_mean[c] = (1.0 - m) * bn->running_mean[c] + m * mean;
            double unbiased_var = (batch_size > 1)
                ? var_sum / (double)(batch_size - 1)
                : var;
            bn->running_var[c]  = (1.0 - m) * bn->running_var[c]  + m * unbiased_var;
        }
    } else {
        /* Eval mode: use running statistics */
        for (int64_t c = 0; c < C; c++) {
            double invstd = 1.0 / sqrt(bn->running_var[c] + eps);
            for (int64_t b = 0; b < batch_size; b++) {
                double xh = (input[b * C + c] - bn->running_mean[c]) * invstd;
                bn->x_hat[b * C + c] = xh;
                output[b * C + c] = bn->gamma[c] * xh + bn->beta[c];
            }
        }
    }

    return 0;
}

/* ---- backward ----------------------------------------------------- */

/*
 * batchnorm1d_backward
 *
 *   grad_output:  [B × C]  dL/dy
 *   grad_input:   [B × C]  dL/dx  (output)
 *   grad_gamma:   [C]      dL/dgamma  (output, accumulated)
 *   grad_beta:    [C]      dL/dbeta   (output, accumulated)
 *
 * Uses saved_mean, saved_invstd, x_hat from the last forward call.
 */
int batchnorm1d_backward(BatchNorm1d *bn,
                          const double *grad_output,
                          double *grad_input,
                          double *grad_gamma,
                          double *grad_beta,
                          int64_t batch_size)
{
    if (!bn || !grad_output || !grad_input || !grad_gamma || !grad_beta)
        return -1;
    if (batch_size <= 0 || !bn->x_hat) return -1;

    int64_t C = bn->num_features;
    double N = (double)batch_size;

    for (int64_t c = 0; c < C; c++) {
        double invstd = bn->saved_invstd[c];
        double g = bn->gamma[c];

        /* Accumulate grad_gamma and grad_beta */
        double dg = 0.0, db = 0.0;
        for (int64_t b = 0; b < batch_size; b++) {
            double dy = grad_output[b * C + c];
            dg += dy * bn->x_hat[b * C + c];
            db += dy;
        }
        grad_gamma[c] = dg;
        grad_beta[c]  = db;

        /* grad_input:
         * dx = (1/N) * gamma * invstd * (N*dy - sum(dy) - x_hat * sum(dy * x_hat))
         */
        double sum_dy = db;
        double sum_dy_xhat = dg;

        for (int64_t b = 0; b < batch_size; b++) {
            double dy = grad_output[b * C + c];
            double xh = bn->x_hat[b * C + c];
            grad_input[b * C + c] = g * invstd / N
                * (N * dy - sum_dy - xh * sum_dy_xhat);
        }
    }

    return 0;
}

/* ---- accessors ---------------------------------------------------- */

double *batchnorm1d_gamma(BatchNorm1d *bn) { return bn ? bn->gamma : NULL; }
double *batchnorm1d_beta(BatchNorm1d *bn)  { return bn ? bn->beta : NULL; }
double *batchnorm1d_running_mean(BatchNorm1d *bn) { return bn ? bn->running_mean : NULL; }
double *batchnorm1d_running_var(BatchNorm1d *bn)  { return bn ? bn->running_var : NULL; }
int64_t batchnorm1d_num_features(BatchNorm1d *bn) { return bn ? bn->num_features : 0; }


/* ================================================================== */
/* LayerNorm                                                          */
/* ================================================================== */

typedef struct {
    int64_t  num_features;
    double   eps;
    double  *gamma;         /* scale, length = num_features */
    double  *beta;          /* shift, length = num_features */
    /* Saved for backward: */
    double  *saved_mean;    /* per-sample mean, length = batch_size  */
    double  *saved_invstd;  /* per-sample invstd, length = batch_size */
    double  *x_hat;         /* normalised input (B*C) */
    int64_t  last_batch;
} LayerNorm;

LayerNorm *layernorm_create(int64_t num_features, double eps)
{
    if (num_features <= 0) return NULL;

    LayerNorm *ln = calloc(1, sizeof(*ln));
    if (!ln) return NULL;

    ln->num_features = num_features;
    ln->eps          = eps;
    ln->gamma        = malloc((size_t)num_features * sizeof(double));
    ln->beta         = calloc((size_t)num_features, sizeof(double));
    ln->saved_mean   = NULL;
    ln->saved_invstd = NULL;
    ln->x_hat        = NULL;
    ln->last_batch   = 0;

    if (!ln->gamma || !ln->beta) {
        free(ln->gamma); free(ln->beta); free(ln);
        return NULL;
    }

    for (int64_t i = 0; i < num_features; i++)
        ln->gamma[i] = 1.0;

    return ln;
}

void layernorm_free(LayerNorm *ln)
{
    if (!ln) return;
    free(ln->gamma); free(ln->beta);
    free(ln->saved_mean); free(ln->saved_invstd);
    free(ln->x_hat);
    free(ln);
}

/*
 * layernorm_forward
 *
 *   input:  [B × C]   output: [B × C]
 *   Normalizes across features (axis=1) for each sample independently.
 */
int layernorm_forward(LayerNorm *ln,
                      const double *input, double *output,
                      int64_t batch_size)
{
    if (!ln || !input || !output || batch_size <= 0) return -1;

    int64_t C = ln->num_features;
    int64_t total = batch_size * C;

    /* Re-allocate saved buffers if batch size changed */
    if (ln->last_batch != batch_size || !ln->x_hat) {
        free(ln->saved_mean);  free(ln->saved_invstd); free(ln->x_hat);
        ln->saved_mean   = malloc((size_t)batch_size * sizeof(double));
        ln->saved_invstd = malloc((size_t)batch_size * sizeof(double));
        ln->x_hat        = malloc((size_t)total * sizeof(double));
        if (!ln->saved_mean || !ln->saved_invstd || !ln->x_hat) return -1;
    }
    ln->last_batch = batch_size;

    for (int64_t b = 0; b < batch_size; b++) {
        const double *row = input + b * C;

        /* Mean */
        double sum = 0.0;
        for (int64_t c = 0; c < C; c++) sum += row[c];
        double mean = sum / (double)C;

        /* Variance */
        double var_sum = 0.0;
        for (int64_t c = 0; c < C; c++) {
            double d = row[c] - mean;
            var_sum += d * d;
        }
        double var = var_sum / (double)C;
        double invstd = 1.0 / sqrt(var + ln->eps);

        ln->saved_mean[b]   = mean;
        ln->saved_invstd[b] = invstd;

        /* Normalize, scale, shift */
        for (int64_t c = 0; c < C; c++) {
            double xh = (row[c] - mean) * invstd;
            ln->x_hat[b * C + c] = xh;
            output[b * C + c] = ln->gamma[c] * xh + ln->beta[c];
        }
    }

    return 0;
}

/*
 * layernorm_backward
 *
 *   grad_output:  [B × C]
 *   grad_input:   [B × C]
 *   grad_gamma:   [C]  (accumulated over batch)
 *   grad_beta:    [C]  (accumulated over batch)
 */
int layernorm_backward(LayerNorm *ln,
                       const double *grad_output,
                       double *grad_input,
                       double *grad_gamma,
                       double *grad_beta,
                       int64_t batch_size)
{
    if (!ln || !grad_output || !grad_input || !grad_gamma || !grad_beta)
        return -1;
    if (batch_size <= 0 || !ln->x_hat) return -1;

    int64_t C = ln->num_features;
    double N = (double)C;

    /* Zero out grad_gamma/beta since we accumulate */
    memset(grad_gamma, 0, (size_t)C * sizeof(double));
    memset(grad_beta,  0, (size_t)C * sizeof(double));

    for (int64_t b = 0; b < batch_size; b++) {
        double invstd = ln->saved_invstd[b];

        /* Accumulate grad_gamma and grad_beta */
        double sum_dy = 0.0, sum_dy_xhat = 0.0;
        for (int64_t c = 0; c < C; c++) {
            double dy = grad_output[b * C + c];
            double xh = ln->x_hat[b * C + c];
            grad_gamma[c] += dy * xh;
            grad_beta[c]  += dy;
            sum_dy       += ln->gamma[c] * dy;
            sum_dy_xhat  += ln->gamma[c] * dy * xh;
        }

        /* grad_input */
        for (int64_t c = 0; c < C; c++) {
            double dy = grad_output[b * C + c];
            double xh = ln->x_hat[b * C + c];
            grad_input[b * C + c] = invstd / N
                * (N * ln->gamma[c] * dy - sum_dy - xh * sum_dy_xhat);
        }
    }

    return 0;
}

double *layernorm_gamma(LayerNorm *ln) { return ln ? ln->gamma : NULL; }
double *layernorm_beta(LayerNorm *ln)  { return ln ? ln->beta : NULL; }
int64_t layernorm_num_features(LayerNorm *ln) { return ln ? ln->num_features : 0; }
