/*
 * conv2d.c — Conv2D and MaxPool2D layers (forward + backward)
 *
 * Part of the Neural-Assembly framework.
 *
 * The convolution uses im2col + GEMM, which is how most serious frameworks
 * do it under the hood.  We unroll the input patches into a column matrix,
 * multiply by the weight matrix, and that gives us the output feature map.
 * Backward is col2im for the input gradient and outer product for the weight
 * gradient.  Not the fanciest approach but it's correct and reasonably fast.
 *
 * MaxPool stores the indices of the max elements during forward so we can
 * route gradients back to the right positions during backward.
 *
 * All data is float64 (double), stored in NCHW order:
 *   [batch, channels, height, width]
 *
 * Functions:
 *   conv2d_layer_create      — allocate a Conv2D layer with Xavier-init weights
 *   conv2d_layer_free        — tear down and free everything
 *   conv2d_layer_forward     — im2col + GEMM forward pass
 *   conv2d_layer_backward    — gradient w.r.t. input, weights, and bias
 *   conv2d_layer_weight      — pointer to the weight buffer
 *   conv2d_layer_bias        — pointer to the bias buffer
 *   conv2d_layer_out_h       — compute output height for given input height
 *   conv2d_layer_out_w       — compute output width for given input width
 *
 *   maxpool2d_forward        — max pooling with index tracking
 *   maxpool2d_backward       — scatter gradients back using stored indices
 *
 *   conv2d_output_size       — pure helper: compute output spatial dim
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>


/* ── Helpers ─────────────────────────────────────────────────────── */

/* Output spatial dimension for conv or pool */
static inline int64_t calc_out_dim(int64_t input_dim, int64_t kernel_dim,
                                   int64_t stride, int64_t padding)
{
    return (input_dim + 2 * padding - kernel_dim) / stride + 1;
}

/* Public version so Python can call it without creating a layer */
int64_t conv2d_output_size(int64_t input_dim, int64_t kernel_dim,
                           int64_t stride, int64_t padding)
{
    if (stride < 1) return -1;
    if (input_dim + 2 * padding < kernel_dim) return -1;
    return calc_out_dim(input_dim, kernel_dim, stride, padding);
}


/* ── Im2col / Col2im ─────────────────────────────────────────────── */

/*
 * im2col takes a single image (C × H × W) and unrolls every receptive
 * field into a column of a matrix.  The result is a matrix of shape:
 *   (C * kH * kW) × (out_h * out_w)
 *
 * This lets us express convolution as a matrix multiply:
 *   output = weight @ col
 * where weight is (out_channels × C*kH*kW).
 */
static void im2col(const double *data_im,
                   int64_t channels, int64_t height, int64_t width,
                   int64_t kh, int64_t kw,
                   int64_t stride, int64_t pad,
                   double *data_col)
{
    int64_t out_h = calc_out_dim(height, kh, stride, pad);
    int64_t out_w = calc_out_dim(width, kw, stride, pad);
    int64_t col_idx = 0;

    for (int64_t c = 0; c < channels; c++) {
        for (int64_t kk_h = 0; kk_h < kh; kk_h++) {
            for (int64_t kk_w = 0; kk_w < kw; kk_w++) {
                for (int64_t oh = 0; oh < out_h; oh++) {
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        int64_t ih = oh * stride - pad + kk_h;
                        int64_t iw = ow * stride - pad + kk_w;
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            data_col[col_idx] = data_im[c * height * width + ih * width + iw];
                        } else {
                            data_col[col_idx] = 0.0;  /* zero-padding */
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}

/*
 * col2im is the inverse: scatter the column matrix back into image
 * space.  Used during backward to accumulate the input gradient.
 * Note: this *accumulates* into data_im, so zero it first.
 */
static void col2im(const double *data_col,
                   int64_t channels, int64_t height, int64_t width,
                   int64_t kh, int64_t kw,
                   int64_t stride, int64_t pad,
                   double *data_im)
{
    int64_t out_h = calc_out_dim(height, kh, stride, pad);
    int64_t out_w = calc_out_dim(width, kw, stride, pad);
    int64_t col_idx = 0;

    for (int64_t c = 0; c < channels; c++) {
        for (int64_t kk_h = 0; kk_h < kh; kk_h++) {
            for (int64_t kk_w = 0; kk_w < kw; kk_w++) {
                for (int64_t oh = 0; oh < out_h; oh++) {
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        int64_t ih = oh * stride - pad + kk_h;
                        int64_t iw = ow * stride - pad + kk_w;
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            data_im[c * height * width + ih * width + iw] += data_col[col_idx];
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}


/* ── Simple GEMM ─────────────────────────────────────────────────── */

/*
 * C = alpha * A @ B + beta * C
 * A is (M × K), B is (K × N), C is (M × N)
 *
 * Not fancy — no tiling or SIMD here — but it's correct and good
 * enough for the kernel sizes we'll typically see.  The assembly
 * SIMD matmul can be swapped in later for bigger workloads.
 */
static void gemm_nn(int64_t M, int64_t N, int64_t K,
                    double alpha,
                    const double *A, int64_t lda,
                    const double *B, int64_t ldb,
                    double beta,
                    double *C, int64_t ldc)
{
    /* Scale C by beta first */
    if (beta == 0.0) {
        for (int64_t i = 0; i < M * ldc; i++) C[i] = 0.0;
    } else if (beta != 1.0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] *= beta;
    }

    /* C += alpha * A @ B */
    for (int64_t i = 0; i < M; i++) {
        for (int64_t k = 0; k < K; k++) {
            double a_ik = alpha * A[i * lda + k];
            for (int64_t j = 0; j < N; j++) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

/* C = alpha * A^T @ B + beta * C    (A is K×M, we want M×N) */
static void gemm_tn(int64_t M, int64_t N, int64_t K,
                    double alpha,
                    const double *A, int64_t lda,
                    const double *B, int64_t ldb,
                    double beta,
                    double *C, int64_t ldc)
{
    if (beta == 0.0) {
        for (int64_t i = 0; i < M * ldc; i++) C[i] = 0.0;
    } else if (beta != 1.0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] *= beta;
    }

    for (int64_t k = 0; k < K; k++) {
        for (int64_t i = 0; i < M; i++) {
            double a_ki = alpha * A[k * lda + i];
            for (int64_t j = 0; j < N; j++) {
                C[i * ldc + j] += a_ki * B[k * ldb + j];
            }
        }
    }
}

/* C = alpha * A @ B^T + beta * C    (B is N×K, we want M×N) */
static void gemm_nt(int64_t M, int64_t N, int64_t K,
                    double alpha,
                    const double *A, int64_t lda,
                    const double *B, int64_t ldb,
                    double beta,
                    double *C, int64_t ldc)
{
    if (beta == 0.0) {
        for (int64_t i = 0; i < M * ldc; i++) C[i] = 0.0;
    } else if (beta != 1.0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] *= beta;
    }

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double dot = 0.0;
            for (int64_t k = 0; k < K; k++) {
                dot += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += alpha * dot;
        }
    }
}


/* ── Conv2D layer struct ─────────────────────────────────────────── */

typedef struct Conv2DLayer {
    int64_t in_channels;
    int64_t out_channels;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride;
    int64_t padding;
    int     has_bias;

    /* weight: out_channels × (in_channels * kernel_h * kernel_w) */
    double *weight;
    /* bias: out_channels (NULL if no bias) */
    double *bias;

    /* Cached from last forward — needed for backward */
    double *col_buf;       /* im2col buffer */
    int64_t col_buf_size;
    int64_t last_batch;
    int64_t last_in_h;
    int64_t last_in_w;
} Conv2DLayer;


/* ── Conv2D lifecycle ────────────────────────────────────────────── */

Conv2DLayer *conv2d_layer_create(int64_t in_channels, int64_t out_channels,
                                 int64_t kernel_h, int64_t kernel_w,
                                 int64_t stride, int64_t padding,
                                 int has_bias)
{
    if (in_channels < 1 || out_channels < 1) return NULL;
    if (kernel_h < 1 || kernel_w < 1) return NULL;
    if (stride < 1) return NULL;
    if (padding < 0) return NULL;

    Conv2DLayer *layer = calloc(1, sizeof(Conv2DLayer));
    if (!layer) return NULL;

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_h = kernel_h;
    layer->kernel_w = kernel_w;
    layer->stride = stride;
    layer->padding = padding;
    layer->has_bias = has_bias;

    /* Allocate weight: out_channels × (in_channels * kH * kW) */
    int64_t wsize = out_channels * in_channels * kernel_h * kernel_w;
    layer->weight = malloc(sizeof(double) * (size_t)wsize);
    if (!layer->weight) { free(layer); return NULL; }

    /* Xavier / Kaiming uniform initialisation.
       fan_in = in_channels * kH * kW, fan_out = out_channels * kH * kW
       We use the Kaiming formula: std = sqrt(2 / fan_in) */
    double fan_in = (double)(in_channels * kernel_h * kernel_w);
    double bound = sqrt(2.0 / fan_in);

    /* Simple LCG PRNG — same approach as embedding.c.
       Not crypto-grade, but good enough for weight init. */
    uint64_t rng = 2654435761ULL ^ (uint64_t)wsize;
    for (int64_t i = 0; i < wsize; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        /* Map to [-bound, +bound] */
        double u = ((double)(rng >> 11) / (double)(1ULL << 53));  /* [0, 1) */
        layer->weight[i] = (2.0 * u - 1.0) * bound;
    }

    /* Allocate bias (initialised to zero) */
    if (has_bias) {
        layer->bias = calloc((size_t)out_channels, sizeof(double));
        if (!layer->bias) {
            free(layer->weight);
            free(layer);
            return NULL;
        }
    }

    return layer;
}

void conv2d_layer_free(Conv2DLayer *layer)
{
    if (!layer) return;
    free(layer->weight);
    free(layer->bias);
    free(layer->col_buf);
    free(layer);
}


/* ── Conv2D forward ──────────────────────────────────────────────── */

int conv2d_layer_forward(Conv2DLayer *layer,
                         const double *input, int64_t batch,
                         int64_t in_h, int64_t in_w,
                         double *output)
{
    /*
     * input:  [batch, in_channels, in_h, in_w]       — row-major NCHW
     * output: [batch, out_channels, out_h, out_w]     — row-major NCHW
     *
     * For each sample in the batch:
     *   1. im2col the input → col of shape (in_c*kH*kW) × (out_h*out_w)
     *   2. output = weight @ col  (GEMM: out_c × spatial_out)
     *   3. add bias if present
     */
    if (!layer || !input || !output) return -1;
    if (batch < 1 || in_h < 1 || in_w < 1) return -1;

    int64_t kh = layer->kernel_h, kw = layer->kernel_w;
    int64_t s  = layer->stride,   p  = layer->padding;
    int64_t ic = layer->in_channels, oc = layer->out_channels;

    int64_t out_h = calc_out_dim(in_h, kh, s, p);
    int64_t out_w = calc_out_dim(in_w, kw, s, p);
    if (out_h < 1 || out_w < 1) return -1;

    int64_t spatial_out = out_h * out_w;
    int64_t col_rows = ic * kh * kw;
    int64_t col_size = col_rows * spatial_out;

    /* (Re)allocate the im2col buffer if needed */
    if (col_size > layer->col_buf_size) {
        free(layer->col_buf);
        layer->col_buf = malloc(sizeof(double) * (size_t)col_size);
        if (!layer->col_buf) { layer->col_buf_size = 0; return -1; }
        layer->col_buf_size = col_size;
    }

    /* Cache for backward */
    layer->last_batch = batch;
    layer->last_in_h = in_h;
    layer->last_in_w = in_w;

    int64_t in_spatial = ic * in_h * in_w;
    int64_t out_spatial = oc * out_h * out_w;

    for (int64_t b = 0; b < batch; b++) {
        const double *in_b  = input  + b * in_spatial;
        double       *out_b = output + b * out_spatial;

        /* Step 1: unroll input patches into columns */
        im2col(in_b, ic, in_h, in_w, kh, kw, s, p, layer->col_buf);

        /* Step 2: GEMM — output[oc × spatial_out] = weight[oc × col_rows] @ col[col_rows × spatial_out] */
        gemm_nn(oc, spatial_out, col_rows,
                1.0,
                layer->weight, col_rows,
                layer->col_buf, spatial_out,
                0.0,
                out_b, spatial_out);

        /* Step 3: add bias — broadcast across spatial dimensions */
        if (layer->has_bias && layer->bias) {
            for (int64_t c = 0; c < oc; c++) {
                double bval = layer->bias[c];
                double *row = out_b + c * spatial_out;
                for (int64_t s_idx = 0; s_idx < spatial_out; s_idx++) {
                    row[s_idx] += bval;
                }
            }
        }
    }

    return 0;
}


/* ── Conv2D backward ─────────────────────────────────────────────── */

int conv2d_layer_backward(Conv2DLayer *layer,
                          const double *input,
                          const double *grad_output,
                          int64_t batch, int64_t in_h, int64_t in_w,
                          double *grad_input,
                          double *grad_weight,
                          double *grad_bias)
{
    /*
     * grad_output: [batch, out_channels, out_h, out_w]
     *
     * Computes:
     *   grad_weight += grad_output_reshaped @ col^T   (for each sample)
     *   grad_bias   += sum over spatial dims of grad_output
     *   grad_input  ← col2im(weight^T @ grad_output_reshaped)
     *
     * All grad buffers must be pre-zeroed by the caller if you want
     * fresh gradients.  Otherwise they accumulate (handy for gradient
     * accumulation across micro-batches).
     */
    if (!layer || !grad_output) return -1;
    if (batch < 1 || in_h < 1 || in_w < 1) return -1;

    int64_t kh = layer->kernel_h, kw = layer->kernel_w;
    int64_t s  = layer->stride,   p  = layer->padding;
    int64_t ic = layer->in_channels, oc = layer->out_channels;

    int64_t out_h = calc_out_dim(in_h, kh, s, p);
    int64_t out_w = calc_out_dim(in_w, kw, s, p);
    int64_t spatial_out = out_h * out_w;
    int64_t col_rows = ic * kh * kw;
    int64_t col_size = col_rows * spatial_out;

    /* We need a scratch col buffer for this */
    double *col_scratch = malloc(sizeof(double) * (size_t)col_size);
    if (!col_scratch) return -1;

    int64_t in_spatial  = ic * in_h * in_w;
    int64_t out_spatial = oc * out_h * out_w;

    for (int64_t b = 0; b < batch; b++) {
        const double *grad_out_b = grad_output + b * out_spatial;

        /* grad_weight: accumulate weight gradients
         * For each sample: grad_weight += grad_out_b @ col_b^T
         * grad_out_b is (oc × spatial_out), col_b is (col_rows × spatial_out)
         * So grad_weight (oc × col_rows) += grad_out_b @ col_b^T
         */
        if (grad_weight && input) {
            const double *in_b = input + b * in_spatial;
            im2col(in_b, ic, in_h, in_w, kh, kw, s, p, col_scratch);
            /* grad_weight[oc × col_rows] += grad_out[oc × spatial_out] @ col[col_rows × spatial_out]^T */
            gemm_nt(oc, col_rows, spatial_out,
                    1.0,
                    grad_out_b, spatial_out,
                    col_scratch, spatial_out,
                    1.0,  /* accumulate */
                    grad_weight, col_rows);
        }

        /* grad_bias: just sum over spatial dimensions for each output channel */
        if (grad_bias) {
            for (int64_t c = 0; c < oc; c++) {
                const double *go = grad_out_b + c * spatial_out;
                for (int64_t s_idx = 0; s_idx < spatial_out; s_idx++) {
                    grad_bias[c] += go[s_idx];
                }
            }
        }

        /* grad_input: weight^T @ grad_out, then col2im */
        if (grad_input) {
            /* col_scratch[col_rows × spatial_out] = weight^T[col_rows × oc] @ grad_out_b[oc × spatial_out] */
            gemm_tn(col_rows, spatial_out, oc,
                    1.0,
                    layer->weight, col_rows,
                    grad_out_b, spatial_out,
                    0.0,
                    col_scratch, spatial_out);

            double *gi_b = grad_input + b * in_spatial;
            col2im(col_scratch, ic, in_h, in_w, kh, kw, s, p, gi_b);
        }
    }

    free(col_scratch);
    return 0;
}


/* ── Conv2D accessors ────────────────────────────────────────────── */

double *conv2d_layer_weight(Conv2DLayer *layer)
{
    return layer ? layer->weight : NULL;
}

double *conv2d_layer_bias(Conv2DLayer *layer)
{
    return layer ? layer->bias : NULL;
}

int64_t conv2d_layer_out_h(const Conv2DLayer *layer, int64_t in_h)
{
    if (!layer) return -1;
    return calc_out_dim(in_h, layer->kernel_h, layer->stride, layer->padding);
}

int64_t conv2d_layer_out_w(const Conv2DLayer *layer, int64_t in_w)
{
    if (!layer) return -1;
    return calc_out_dim(in_w, layer->kernel_w, layer->stride, layer->padding);
}

int64_t conv2d_layer_weight_size(const Conv2DLayer *layer)
{
    if (!layer) return 0;
    return layer->out_channels * layer->in_channels *
           layer->kernel_h * layer->kernel_w;
}

int64_t conv2d_layer_in_channels(const Conv2DLayer *layer)
{
    return layer ? layer->in_channels : 0;
}

int64_t conv2d_layer_out_channels(const Conv2DLayer *layer)
{
    return layer ? layer->out_channels : 0;
}

int64_t conv2d_layer_kernel_h(const Conv2DLayer *layer)
{
    return layer ? layer->kernel_h : 0;
}

int64_t conv2d_layer_kernel_w(const Conv2DLayer *layer)
{
    return layer ? layer->kernel_w : 0;
}

int64_t conv2d_layer_stride(const Conv2DLayer *layer)
{
    return layer ? layer->stride : 0;
}

int64_t conv2d_layer_padding(const Conv2DLayer *layer)
{
    return layer ? layer->padding : 0;
}


/* ── MaxPool2D ───────────────────────────────────────────────────── */

/*
 * MaxPool doesn't need a persistent struct — it's stateless apart from
 * the index mask used during backward.  We provide a simple functional
 * API: forward writes output + mask, backward uses the mask.
 */

int maxpool2d_forward(const double *input,
                      int64_t batch, int64_t channels,
                      int64_t in_h, int64_t in_w,
                      int64_t pool_h, int64_t pool_w,
                      int64_t stride, int64_t padding,
                      double *output, int64_t *mask)
{
    /*
     * input:  [batch, channels, in_h, in_w]
     * output: [batch, channels, out_h, out_w]
     * mask:   [batch, channels, out_h, out_w]  — index of max element
     *         in the flattened (in_h × in_w) space, for backward routing.
     */
    if (!input || !output) return -1;
    if (batch < 1 || channels < 1 || in_h < 1 || in_w < 1) return -1;
    if (pool_h < 1 || pool_w < 1 || stride < 1 || padding < 0) return -1;

    int64_t out_h = calc_out_dim(in_h, pool_h, stride, padding);
    int64_t out_w = calc_out_dim(in_w, pool_w, stride, padding);
    if (out_h < 1 || out_w < 1) return -1;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            const double *in_plane  = input  + (b * channels + c) * in_h * in_w;
            double       *out_plane = output + (b * channels + c) * out_h * out_w;
            int64_t      *msk_plane = mask ? (mask + (b * channels + c) * out_h * out_w) : NULL;

            for (int64_t oh = 0; oh < out_h; oh++) {
                for (int64_t ow = 0; ow < out_w; ow++) {
                    double max_val = -DBL_MAX;
                    int64_t max_idx = -1;

                    for (int64_t kh = 0; kh < pool_h; kh++) {
                        for (int64_t kw_i = 0; kw_i < pool_w; kw_i++) {
                            int64_t ih = oh * stride - padding + kh;
                            int64_t iw = ow * stride - padding + kw_i;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                double val = in_plane[ih * in_w + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = ih * in_w + iw;
                                }
                            }
                        }
                    }

                    out_plane[oh * out_w + ow] = (max_idx >= 0) ? max_val : 0.0;
                    if (msk_plane) {
                        msk_plane[oh * out_w + ow] = max_idx;
                    }
                }
            }
        }
    }

    return 0;
}

int maxpool2d_backward(const double *grad_output,
                       const int64_t *mask,
                       int64_t batch, int64_t channels,
                       int64_t in_h, int64_t in_w,
                       int64_t out_h, int64_t out_w,
                       double *grad_input)
{
    /*
     * Routes each element of grad_output back to where the max came
     * from (stored in mask).  grad_input must be pre-zeroed.
     */
    if (!grad_output || !mask || !grad_input) return -1;
    if (batch < 1 || channels < 1) return -1;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            const double  *go = grad_output + (b * channels + c) * out_h * out_w;
            const int64_t *mk = mask        + (b * channels + c) * out_h * out_w;
            double        *gi = grad_input  + (b * channels + c) * in_h * in_w;

            for (int64_t i = 0; i < out_h * out_w; i++) {
                int64_t idx = mk[i];
                if (idx >= 0 && idx < in_h * in_w) {
                    gi[idx] += go[i];
                }
            }
        }
    }

    return 0;
}
