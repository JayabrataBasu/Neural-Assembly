/*
 * transforms.c — Data transforms: Normalize, Standardize (min-max), per-feature
 *
 * Part of the Neural-Assembly framework.
 * All operate on double* arrays (float64) for maximum precision.
 *
 * Functions:
 *   transform_normalize     — z-score: (x - mean) / std, per feature
 *   transform_unnormalize   — inverse of z-score
 *   transform_minmax        — scale to [0, 1] per feature
 *   transform_compute_stats — compute mean/std/min/max from data
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ───────────────────────────────────────────────────────────────────
 *  compute per-feature statistics from data[batch_size × num_features]
 *
 *  Outputs: mean[num_features], std[num_features],
 *           min_val[num_features], max_val[num_features]
 *  Any output pointer may be NULL (that stat is skipped).
 * ─────────────────────────────────────────────────────────────────── */
int transform_compute_stats(const double *data,
                            int64_t batch_size,
                            int64_t num_features,
                            double *mean_out,
                            double *std_out,
                            double *min_out,
                            double *max_out)
{
    if (!data || batch_size <= 0 || num_features <= 0)
        return -1;

    for (int64_t f = 0; f < num_features; f++) {
        double sum  = 0.0;
        double sum2 = 0.0;
        double mn   =  1e308;
        double mx   = -1e308;

        for (int64_t i = 0; i < batch_size; i++) {
            double v = data[i * num_features + f];
            sum  += v;
            sum2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        double m = sum / (double)batch_size;
        /* population std (not Bessel-corrected) */
        double var = sum2 / (double)batch_size - m * m;
        if (var < 0.0) var = 0.0;  /* numerical guard */
        double s = sqrt(var);

        if (mean_out) mean_out[f] = m;
        if (std_out)  std_out[f]  = s;
        if (min_out)  min_out[f]  = mn;
        if (max_out)  max_out[f]  = mx;
    }
    return 0;
}


/* ───────────────────────────────────────────────────────────────────
 *  Z-score normalisation:  out[i][f] = (data[i][f] - mean[f]) / (std[f] + eps)
 * ─────────────────────────────────────────────────────────────────── */
int transform_normalize(const double *data,
                        double *output,
                        int64_t batch_size,
                        int64_t num_features,
                        const double *mean,
                        const double *std,
                        double eps)
{
    if (!data || !output || !mean || !std)
        return -1;
    if (batch_size <= 0 || num_features <= 0)
        return -1;

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t f = 0; f < num_features; f++) {
            int64_t idx = i * num_features + f;
            output[idx] = (data[idx] - mean[f]) / (std[f] + eps);
        }
    }
    return 0;
}


/* ───────────────────────────────────────────────────────────────────
 *  Inverse z-score:  out[i][f] = data[i][f] * (std[f] + eps) + mean[f]
 * ─────────────────────────────────────────────────────────────────── */
int transform_unnormalize(const double *data,
                          double *output,
                          int64_t batch_size,
                          int64_t num_features,
                          const double *mean,
                          const double *std,
                          double eps)
{
    if (!data || !output || !mean || !std)
        return -1;
    if (batch_size <= 0 || num_features <= 0)
        return -1;

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t f = 0; f < num_features; f++) {
            int64_t idx = i * num_features + f;
            output[idx] = data[idx] * (std[f] + eps) + mean[f];
        }
    }
    return 0;
}


/* ───────────────────────────────────────────────────────────────────
 *  Min-max scaling:  out[i][f] = (data[i][f] - min[f]) / (max[f] - min[f] + eps)
 *  Maps data to [0, 1] per feature.
 * ─────────────────────────────────────────────────────────────────── */
int transform_minmax(const double *data,
                     double *output,
                     int64_t batch_size,
                     int64_t num_features,
                     const double *min_val,
                     const double *max_val,
                     double eps)
{
    if (!data || !output || !min_val || !max_val)
        return -1;
    if (batch_size <= 0 || num_features <= 0)
        return -1;

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t f = 0; f < num_features; f++) {
            int64_t idx = i * num_features + f;
            double range = max_val[f] - min_val[f] + eps;
            output[idx] = (data[idx] - min_val[f]) / range;
        }
    }
    return 0;
}
