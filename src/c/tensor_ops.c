/*
 * tensor_ops.c — Tensor utility kernels (float64)
 *
 * Batch 12 feature set:
 *   - concat (2D axis 0/1)
 *   - split  (2D axis 0/1)
 *   - pad    (2D constant pad)
 *   - transpose (2D)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int tensor_concat_2d(const double *a,
                     const double *b,
                     int64_t rows_a, int64_t cols_a,
                     int64_t rows_b, int64_t cols_b,
                     int64_t axis,
                     double *out,
                     int64_t *out_rows,
                     int64_t *out_cols)
{
    if (!a || !b || !out || !out_rows || !out_cols) return -1;
    if (rows_a <= 0 || cols_a <= 0 || rows_b <= 0 || cols_b <= 0) return -1;
    if (axis != 0 && axis != 1) return -1;

    if (axis == 0) {
        if (cols_a != cols_b) return -1;
        *out_rows = rows_a + rows_b;
        *out_cols = cols_a;

        memcpy(out, a, (size_t)rows_a * (size_t)cols_a * sizeof(double));
        memcpy(out + rows_a * cols_a,
               b,
               (size_t)rows_b * (size_t)cols_b * sizeof(double));
        return 0;
    }

    if (rows_a != rows_b) return -1;
    *out_rows = rows_a;
    *out_cols = cols_a + cols_b;

    for (int64_t r = 0; r < rows_a; ++r) {
        double *dst = out + r * (*out_cols);
        const double *src_a = a + r * cols_a;
        const double *src_b = b + r * cols_b;

        memcpy(dst, src_a, (size_t)cols_a * sizeof(double));
        memcpy(dst + cols_a, src_b, (size_t)cols_b * sizeof(double));
    }

    return 0;
}

int tensor_split_2d(const double *in,
                    int64_t rows, int64_t cols,
                    int64_t axis,
                    int64_t split_index,
                    double *out_a,
                    double *out_b,
                    int64_t *rows_a,
                    int64_t *cols_a,
                    int64_t *rows_b,
                    int64_t *cols_b)
{
    if (!in || !out_a || !out_b || !rows_a || !cols_a || !rows_b || !cols_b) return -1;
    if (rows <= 0 || cols <= 0) return -1;
    if (axis != 0 && axis != 1) return -1;

    if (axis == 0) {
        if (split_index <= 0 || split_index >= rows) return -1;

        *rows_a = split_index;
        *cols_a = cols;
        *rows_b = rows - split_index;
        *cols_b = cols;

        memcpy(out_a, in, (size_t)(*rows_a) * (size_t)cols * sizeof(double));
        memcpy(out_b,
               in + split_index * cols,
               (size_t)(*rows_b) * (size_t)cols * sizeof(double));
        return 0;
    }

    if (split_index <= 0 || split_index >= cols) return -1;

    *rows_a = rows;
    *cols_a = split_index;
    *rows_b = rows;
    *cols_b = cols - split_index;

    for (int64_t r = 0; r < rows; ++r) {
        const double *src = in + r * cols;
        memcpy(out_a + r * (*cols_a), src, (size_t)(*cols_a) * sizeof(double));
        memcpy(out_b + r * (*cols_b), src + split_index, (size_t)(*cols_b) * sizeof(double));
    }

    return 0;
}

int tensor_pad_2d(const double *in,
                  int64_t in_rows,
                  int64_t in_cols,
                  int64_t pad_top,
                  int64_t pad_bottom,
                  int64_t pad_left,
                  int64_t pad_right,
                  double pad_value,
                  double *out,
                  int64_t *out_rows,
                  int64_t *out_cols)
{
    if (!in || !out || !out_rows || !out_cols) return -1;
    if (in_rows <= 0 || in_cols <= 0) return -1;
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) return -1;

    *out_rows = in_rows + pad_top + pad_bottom;
    *out_cols = in_cols + pad_left + pad_right;
    if (*out_rows <= 0 || *out_cols <= 0) return -1;

    const int64_t total = (*out_rows) * (*out_cols);
    for (int64_t i = 0; i < total; ++i) out[i] = pad_value;

    for (int64_t r = 0; r < in_rows; ++r) {
        double *dst = out + (r + pad_top) * (*out_cols) + pad_left;
        const double *src = in + r * in_cols;
        memcpy(dst, src, (size_t)in_cols * sizeof(double));
    }

    return 0;
}

int tensor_transpose2d_array(const double *in,
                             int64_t rows,
                             int64_t cols,
                             double *out)
{
    if (!in || !out) return -1;
    if (rows <= 0 || cols <= 0) return -1;

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
    return 0;
}
