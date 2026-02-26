/* ============================================================================
 * pruning.c — Magnitude-Based Weight Pruning
 * ============================================================================
 * Provides unstructured and structured pruning for neural network weights.
 *
 * Unstructured: zero out individual weights whose |value| < threshold!
 * Structured:   zero out entire rows/columns if their L2 norm < threshold!
 *
 * The heavy math (threshold scanning, norm computation) can leverage the
 * assembly kernels in training_ops.asm when available, but this C layer
 * handles the control flow, memory management, and mask bookkeeping.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- Unstructured (magnitude) pruning ------------------------------------ */

/*
 * prune_magnitude — zero out weights with |w| < threshold.
 *   weights:   float64 array (read/write, pruned in-place)
 *   mask:      uint8 array (output, 1 = kept, 0 = pruned; may be NULL)
 *   n:         number of elements
 *   threshold: pruning threshold (absolute value)
 *   Returns:   number of weights pruned (zeroed)
 */
int64_t prune_magnitude(double *weights, uint8_t *mask, int64_t n, double threshold) {
    if (!weights || n <= 0 || threshold < 0.0) return -1;

    int64_t pruned = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabs(weights[i]) < threshold) {
            weights[i] = 0.0;
            if (mask) mask[i] = 0;
            pruned++;
        } else {
            if (mask) mask[i] = 1;
        }
    }
    return pruned;
}

/* Comparator helpers (must be at file scope for standard C) */
typedef struct { double absval; int64_t idx; } AbsIdx;

static int cmp_absidx(const void *a, const void *b) {
    double da = ((const AbsIdx *)a)->absval;
    double db = ((const AbsIdx *)b)->absval;
    return (da > db) - (da < db);
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

/*
 * prune_topk — keep only the top-k largest weights by magnitude.
 *   weights:   float64 array (read/write)
 *   mask:      uint8 array (output, may be NULL)
 *   n:         number of elements
 *   keep_ratio: fraction of weights to keep (0.0 .. 1.0)
 *   Returns:   number pruned
 *
 * Strategy: find the k-th largest |weight| via partial sort, then prune.
 */
int64_t prune_topk(double *weights, uint8_t *mask, int64_t n, double keep_ratio) {
    if (!weights || n <= 0 || keep_ratio < 0.0 || keep_ratio > 1.0) return -1;

    int64_t keep_count = (int64_t)(n * keep_ratio + 0.5);
    if (keep_count >= n) {
        /* Nothing to prune */
        if (mask) memset(mask, 1, (size_t)n);
        return 0;
    }
    if (keep_count <= 0) {
        /* Prune everything */
        memset(weights, 0, (size_t)n * sizeof(double));
        if (mask) memset(mask, 0, (size_t)n);
        return n;
    }

    /* Build array of absolute values + indices */
    AbsIdx *arr = (AbsIdx *)malloc((size_t)n * sizeof(AbsIdx));
    if (!arr) return -1;

    for (int64_t i = 0; i < n; i++) {
        arr[i].absval = fabs(weights[i]);
        arr[i].idx = i;
    }

    /* Partial sort: we need the (n - keep_count)-th smallest by absval.
     * Simple approach: use qsort then pick threshold.  For large tensors (I thank DAA which taught me this)
     * a quickselect would be O(n) but qsort is fine here, probably. */
    qsort(arr, (size_t)n, sizeof(AbsIdx), cmp_absidx);

    /* Prune the smallest (n - keep_count) */
    int64_t prune_count = n - keep_count;
    for (int64_t i = 0; i < prune_count; i++) {
        weights[arr[i].idx] = 0.0;
        if (mask) mask[arr[i].idx] = 0;
    }
    for (int64_t i = prune_count; i < n; i++) {
        if (mask) mask[arr[i].idx] = 1;
    }

    free(arr);
    return prune_count;
}

/* ---- Structured pruning -------------------------------------------------- */

/*
 * prune_rows — zero out entire rows whose L2 norm < threshold.
 *   weights:   float64 matrix in row-major order (rows x cols)
 *   rows:      number of rows (e.g., output neurons)
 *   cols:      number of columns (e.g., input features)
 *   threshold: L2 norm threshold for row pruning
 *   row_mask:  uint8 array of size `rows` (output, 1=kept, 0=pruned; may be NULL)
 *   Returns:   number of rows pruned
 */
int64_t prune_rows(double *weights, int64_t rows, int64_t cols,
                   double threshold, uint8_t *row_mask) {
    if (!weights || rows <= 0 || cols <= 0 || threshold < 0.0) return -1;

    int64_t pruned = 0;
    for (int64_t r = 0; r < rows; r++) {
        double norm = 0.0;
        double *row_ptr = weights + r * cols;
        for (int64_t c = 0; c < cols; c++) {
            norm += row_ptr[c] * row_ptr[c];
        }
        norm = sqrt(norm);

        if (norm < threshold) {
            memset(row_ptr, 0, (size_t)cols * sizeof(double));
            if (row_mask) row_mask[r] = 0;
            pruned++;
        } else {
            if (row_mask) row_mask[r] = 1;
        }
    }
    return pruned;
}

/*
 * prune_cols — zero out entire columns whose L2 norm < threshold.
 *   weights:   float64 matrix in row-major order (rows x cols)
 *   rows:      number of rows
 *   cols:      number of columns
 *   threshold: L2 norm threshold for column pruning
 *   col_mask:  uint8 array of size `cols` (output; may be NULL)
 *   Returns:   number of columns pruned
 */
int64_t prune_cols(double *weights, int64_t rows, int64_t cols,
                   double threshold, uint8_t *col_mask) {
    if (!weights || rows <= 0 || cols <= 0 || threshold < 0.0) return -1;

    int64_t pruned = 0;
    for (int64_t c = 0; c < cols; c++) {
        double norm = 0.0;
        for (int64_t r = 0; r < rows; r++) {
            double v = weights[r * cols + c];
            norm += v * v;
        }
        norm = sqrt(norm);

        if (norm < threshold) {
            for (int64_t r = 0; r < rows; r++)
                weights[r * cols + c] = 0.0;
            if (col_mask) col_mask[c] = 0;
            pruned++;
        } else {
            if (col_mask) col_mask[c] = 1;
        }
    }
    return pruned;
}

/* ---- Statistics ---------------------------------------------------------- */

/*
 * compute_sparsity — fraction of zero elements in an array.
 */
double compute_sparsity(const double *weights, int64_t n) {
    if (!weights || n <= 0) return 0.0;
    int64_t zeros = 0;
    for (int64_t i = 0; i < n; i++) {
        if (weights[i] == 0.0) zeros++;
    }
    return (double)zeros / (double)n;
}

/*
 * count_nonzero — count non-zero elements.
 */
int64_t count_nonzero(const double *weights, int64_t n) {
    if (!weights || n <= 0) return 0;
    int64_t nz = 0;
    for (int64_t i = 0; i < n; i++) {
        if (weights[i] != 0.0) nz++;
    }
    return nz;
}

/*
 * compute_threshold_for_sparsity — find the magnitude threshold that
 * achieves approximately the target sparsity ratio.
 *   weights:   float64 array
 *   n:         number of elements
 *   target_sparsity: desired fraction of zeros (0.0 .. 1.0)
 *   Returns:   threshold value
 */
double compute_threshold_for_sparsity(const double *weights, int64_t n,
                                      double target_sparsity) {
    if (!weights || n <= 0 || target_sparsity <= 0.0) return 0.0;
    if (target_sparsity >= 1.0) return 1e30;

    /* Sort absolute values to find the right cutoff */
    double *absvals = (double *)malloc((size_t)n * sizeof(double));
    if (!absvals) return 0.0;

    for (int64_t i = 0; i < n; i++)
        absvals[i] = fabs(weights[i]);

    qsort(absvals, (size_t)n, sizeof(double), cmp_double_asc);

    int64_t cut_idx = (int64_t)(n * target_sparsity);
    if (cut_idx >= n) cut_idx = n - 1;
    double threshold = absvals[cut_idx];

    free(absvals);
    return threshold;
}

/*
 * apply_mask — re-apply a pruning mask (e.g. after an optimizer step).
 *   weights: float64 array (read/write)
 *   mask:    uint8 array (1=keep, 0=zero)
 *   n:       number of elements
 */
void apply_mask(double *weights, const uint8_t *mask, int64_t n) {
    if (!weights || !mask || n <= 0) return;
    for (int64_t i = 0; i < n; i++) {
        if (!mask[i]) weights[i] = 0.0;
    }
}
