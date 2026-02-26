/* ============================================================================
 * quantize.c — INT8 Quantization with Calibration
 * ============================================================================
 * Provides affine (asymmetric) and symmetric quantization of float64/float32
 * weight/activation tensors to INT8, plus dequantization and a basic
 * quantised matrix multiply.
 *
 * Quantization formula (affine):
 *   q = clamp(round(x / scale + zero_point), -128, 127)
 *   x_approx = (q - zero_point) * scale
 *
 * Symmetric:
 *   scale = max(|x|) / 127
 *   q = clamp(round(x / scale), -128, 127)
 *   zero_point = 0
 *
 * Calibration modes:
 *   - MinMax:      scale from observed min/max
 *   - Percentile:  clip to [p, 1-p] percentile to reduce outlier effects
 *   - MovingAvg:   exponential moving average of min/max across batches
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* File-scope comparator for qsort */
static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

/* ---- Quantization parameters --------------------------------------------- */

typedef struct {
    double scale;
    int32_t zero_point;
    double min_val;    /* observed/calibrated minimum */
    double max_val;    /* observed/calibrated maximum */
    int    symmetric;  /* 1 = symmetric, 0 = affine */
} QuantParams;

/* ---- Calibration --------------------------------------------------------- */

/*
 * calibrate_minmax — compute quantization params from raw min/max.
 */
int calibrate_minmax(const double *data, int64_t n, int symmetric,
                     QuantParams *out) {
    if (!data || n <= 0 || !out) return -1;

    double mn = data[0], mx = data[0];
    for (int64_t i = 1; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }

    if (symmetric) {
        double abs_max = fmax(fabs(mn), fabs(mx));
        if (abs_max < 1e-30) abs_max = 1e-30;
        out->scale = abs_max / 127.0;
        out->zero_point = 0;
        out->min_val = -abs_max;
        out->max_val =  abs_max;
        out->symmetric = 1;
    } else {
        if (mn == mx) { mn -= 0.5; mx += 0.5; }
        /* Affine: map [min, max] → [-128, 127] */
        out->scale = (mx - mn) / 255.0;
        if (out->scale < 1e-30) out->scale = 1e-30;
        out->zero_point = (int32_t)round(-128.0 - mn / out->scale);
        /* Clamp zero_point to valid range */
        if (out->zero_point < -128) out->zero_point = -128;
        if (out->zero_point >  127) out->zero_point =  127;
        out->min_val = mn;
        out->max_val = mx;
        out->symmetric = 0;
    }
    return 0;
}

/*
 * calibrate_percentile — clip outliers to [p, 1-p] percentile then minmax.
 */
int calibrate_percentile(const double *data, int64_t n, double percentile,
                         int symmetric, QuantParams *out) {
    if (!data || n <= 0 || !out) return -1;
    if (percentile < 0.0 || percentile > 0.5) percentile = 0.01;

    /* Sort a copy */
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return -1;
    memcpy(sorted, data, (size_t)n * sizeof(double));

    qsort(sorted, (size_t)n, sizeof(double), cmp_double_asc);

    int64_t lo_idx = (int64_t)(n * percentile);
    int64_t hi_idx = (int64_t)(n * (1.0 - percentile));
    if (lo_idx < 0) lo_idx = 0;
    if (hi_idx >= n) hi_idx = n - 1;

    /* Build clipped data for calibration */
    double clipped_min = sorted[lo_idx];
    double clipped_max = sorted[hi_idx];
    free(sorted);

    /* Now calibrate with the clipped range */
    if (symmetric) {
        double abs_max = fmax(fabs(clipped_min), fabs(clipped_max));
        if (abs_max < 1e-30) abs_max = 1e-30;
        out->scale = abs_max / 127.0;
        out->zero_point = 0;
        out->min_val = -abs_max;
        out->max_val =  abs_max;
        out->symmetric = 1;
    } else {
        if (clipped_min == clipped_max) { clipped_min -= 0.5; clipped_max += 0.5; }
        out->scale = (clipped_max - clipped_min) / 255.0;
        if (out->scale < 1e-30) out->scale = 1e-30;
        out->zero_point = (int32_t)round(-128.0 - clipped_min / out->scale);
        if (out->zero_point < -128) out->zero_point = -128;
        if (out->zero_point >  127) out->zero_point =  127;
        out->min_val = clipped_min;
        out->max_val = clipped_max;
        out->symmetric = 0;
    }
    return 0;
}

/* ---- Quantize / Dequantize ----------------------------------------------- */

/*
 * quantize_tensor — convert float64 array to int8 using given params.
 *   data:   input float64 array
 *   out:    output int8 array (must be pre-allocated, size n)
 *   n:      number of elements
 *   params: quantization parameters
 *   Returns 0 on success.
 */
int quantize_tensor(const double *data, int8_t *out, int64_t n,
                    const QuantParams *params) {
    if (!data || !out || n <= 0 || !params) return -1;

    double inv_scale = 1.0 / params->scale;
    int32_t zp = params->zero_point;

    for (int64_t i = 0; i < n; i++) {
        int32_t q = (int32_t)round(data[i] * inv_scale) + zp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
    return 0;
}

/*
 * dequantize_tensor — convert int8 array back to float64.
 */
int dequantize_tensor(const int8_t *data, double *out, int64_t n,
                      const QuantParams *params) {
    if (!data || !out || n <= 0 || !params) return -1;

    double scale = params->scale;
    int32_t zp = params->zero_point;

    for (int64_t i = 0; i < n; i++) {
        out[i] = ((double)data[i] - (double)zp) * scale;
    }
    return 0;
}

/* ---- Quantized matrix multiply ------------------------------------------- */

/*
 * quantized_matmul — int8 matrix multiply with float64 output.
 *   A:     int8 matrix (M x K)
 *   B:     int8 matrix (K x N)
 *   C:     float64 output matrix (M x N)
 *   M, K, N: dimensions
 *   params_a, params_b: quantization params for A and B
 *
 * Computes: C[i][j] = sum_k (dequant(A[i][k]) * dequant(B[k][j]))
 *         = sa * sb * sum_k ((A[i][k] - za) * (B[k][j] - zb))
 *
 * The inner sum is done in int32 to avoid overflow, then scaled.
 */
int quantized_matmul(const int8_t *A, const int8_t *B, double *C,
                     int64_t M, int64_t K, int64_t N,
                     const QuantParams *params_a,
                     const QuantParams *params_b) {
    if (!A || !B || !C || !params_a || !params_b) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -1;

    double combined_scale = params_a->scale * params_b->scale;
    int32_t za = params_a->zero_point;
    int32_t zb = params_b->zero_point;

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; k++) {
                int32_t a_val = (int32_t)A[i * K + k] - za;
                int32_t b_val = (int32_t)B[k * N + j] - zb;
                acc += a_val * b_val;
            }
            C[i * N + j] = (double)acc * combined_scale;
        }
    }
    return 0;
}

/* ---- Quantization error analysis ----------------------------------------- */

/*
 * quantization_error — compute MSE between original and quantized-then-
 * dequantized values.  Useful for calibration validation.
 */
double quantization_error(const double *original, int64_t n,
                          const QuantParams *params) {
    if (!original || n <= 0 || !params) return -1.0;

    double inv_scale = 1.0 / params->scale;
    int32_t zp = params->zero_point;
    double scale = params->scale;
    double mse = 0.0;

    for (int64_t i = 0; i < n; i++) {
        /* Quantize */
        int32_t q = (int32_t)round(original[i] * inv_scale) + zp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        /* Dequantize */
        double recon = ((double)q - (double)zp) * scale;
        double diff = original[i] - recon;
        mse += diff * diff;
    }
    return mse / (double)n;
}

/*
 * quantization_snr — signal-to-noise ratio in dB.
 */
double quantization_snr(const double *original, int64_t n,
                        const QuantParams *params) {
    if (!original || n <= 0 || !params) return -1.0;

    double signal_power = 0.0;
    double noise_power = 0.0;
    double inv_scale = 1.0 / params->scale;
    int32_t zp = params->zero_point;
    double scale = params->scale;

    for (int64_t i = 0; i < n; i++) {
        signal_power += original[i] * original[i];
        int32_t q = (int32_t)round(original[i] * inv_scale) + zp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        double recon = ((double)q - (double)zp) * scale;
        double diff = original[i] - recon;
        noise_power += diff * diff;
    }

    if (noise_power < 1e-30) return 999.0;  /* essentially no noise */
    return 10.0 * log10(signal_power / noise_power);
}
