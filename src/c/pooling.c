/*
 * pooling.c — AvgPool2D + Upsample kernels (float64, NCHW)
 *
 * Batch 11 feature set:
 *   - avgpool2d_forward
 *   - avgpool2d_backward
 *   - upsample2d_nearest_forward
 *   - upsample2d_nearest_backward
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline int64_t out_dim_2d(int64_t input, int64_t kernel, int64_t stride, int64_t pad)
{
    return (input + 2 * pad - kernel) / stride + 1;
}

int avgpool2d_forward(const double *input,
                      int64_t batch, int64_t channels,
                      int64_t in_h, int64_t in_w,
                      int64_t pool_h, int64_t pool_w,
                      int64_t stride, int64_t padding,
                      double *output)
{
    if (!input || !output) return -1;
    if (batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0) return -1;
    if (pool_h <= 0 || pool_w <= 0 || stride <= 0 || padding < 0) return -1;
    if (in_h + 2 * padding < pool_h || in_w + 2 * padding < pool_w) return -1;

    const int64_t out_h = out_dim_2d(in_h, pool_h, stride, padding);
    const int64_t out_w = out_dim_2d(in_w, pool_w, stride, padding);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const double *in_bc = input + (b * channels + c) * in_h * in_w;
            double *out_bc = output + (b * channels + c) * out_h * out_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t h_start = oh * stride - padding;
                    const int64_t w_start = ow * stride - padding;
                    const int64_t h_end = h_start + pool_h;
                    const int64_t w_end = w_start + pool_w;

                    double sum = 0.0;
                    int64_t count = 0;

                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        if (ih < 0 || ih >= in_h) continue;
                        for (int64_t iw = w_start; iw < w_end; ++iw) {
                            if (iw < 0 || iw >= in_w) continue;
                            sum += in_bc[ih * in_w + iw];
                            ++count;
                        }
                    }

                    out_bc[oh * out_w + ow] = (count > 0) ? (sum / (double)count) : 0.0;
                }
            }
        }
    }

    return 0;
}

int avgpool2d_backward(const double *grad_output,
                       int64_t batch, int64_t channels,
                       int64_t in_h, int64_t in_w,
                       int64_t pool_h, int64_t pool_w,
                       int64_t stride, int64_t padding,
                       double *grad_input)
{
    if (!grad_output || !grad_input) return -1;
    if (batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0) return -1;
    if (pool_h <= 0 || pool_w <= 0 || stride <= 0 || padding < 0) return -1;
    if (in_h + 2 * padding < pool_h || in_w + 2 * padding < pool_w) return -1;

    const int64_t out_h = out_dim_2d(in_h, pool_h, stride, padding);
    const int64_t out_w = out_dim_2d(in_w, pool_w, stride, padding);

    memset(grad_input, 0, (size_t)batch * (size_t)channels * (size_t)in_h * (size_t)in_w * sizeof(double));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const double *go_bc = grad_output + (b * channels + c) * out_h * out_w;
            double *gi_bc = grad_input + (b * channels + c) * in_h * in_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t h_start = oh * stride - padding;
                    const int64_t w_start = ow * stride - padding;
                    const int64_t h_end = h_start + pool_h;
                    const int64_t w_end = w_start + pool_w;

                    int64_t count = 0;
                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        if (ih < 0 || ih >= in_h) continue;
                        for (int64_t iw = w_start; iw < w_end; ++iw) {
                            if (iw < 0 || iw >= in_w) continue;
                            ++count;
                        }
                    }

                    if (count <= 0) continue;
                    const double g = go_bc[oh * out_w + ow] / (double)count;

                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        if (ih < 0 || ih >= in_h) continue;
                        for (int64_t iw = w_start; iw < w_end; ++iw) {
                            if (iw < 0 || iw >= in_w) continue;
                            gi_bc[ih * in_w + iw] += g;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int upsample2d_nearest_forward(const double *input,
                               int64_t batch, int64_t channels,
                               int64_t in_h, int64_t in_w,
                               int64_t scale_h, int64_t scale_w,
                               double *output)
{
    if (!input || !output) return -1;
    if (batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0) return -1;
    if (scale_h <= 0 || scale_w <= 0) return -1;

    const int64_t out_h = in_h * scale_h;
    const int64_t out_w = in_w * scale_w;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const double *in_bc = input + (b * channels + c) * in_h * in_w;
            double *out_bc = output + (b * channels + c) * out_h * out_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih = oh / scale_h;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t iw = ow / scale_w;
                    out_bc[oh * out_w + ow] = in_bc[ih * in_w + iw];
                }
            }
        }
    }

    return 0;
}

int upsample2d_nearest_backward(const double *grad_output,
                                int64_t batch, int64_t channels,
                                int64_t in_h, int64_t in_w,
                                int64_t scale_h, int64_t scale_w,
                                double *grad_input)
{
    if (!grad_output || !grad_input) return -1;
    if (batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0) return -1;
    if (scale_h <= 0 || scale_w <= 0) return -1;

    const int64_t out_h = in_h * scale_h;
    const int64_t out_w = in_w * scale_w;

    memset(grad_input, 0, (size_t)batch * (size_t)channels * (size_t)in_h * (size_t)in_w * sizeof(double));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const double *go_bc = grad_output + (b * channels + c) * out_h * out_w;
            double *gi_bc = grad_input + (b * channels + c) * in_h * in_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih = oh / scale_h;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t iw = ow / scale_w;
                    gi_bc[ih * in_w + iw] += go_bc[oh * out_w + ow];
                }
            }
        }
    }

    return 0;
}
