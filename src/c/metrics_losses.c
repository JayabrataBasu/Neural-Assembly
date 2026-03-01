/*
 * metrics_losses.c — Label-smoothed cross-entropy loss + ROC-AUC metric
 *
 * Part of the Neural-Assembly framework.
 * Provides C implementations for:
 *   1. Label-smoothed cross-entropy (forward + backward)
 *   2. ROC-AUC score (binary classification)
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ───────────────────────────────────────────────────────────────────
 *  Label-Smoothed Cross-Entropy Loss
 *
 *  Given raw logits (batch_size × num_classes) and integer targets:
 *
 *    smoothed_target[i][c] = (1 - α) * one_hot(target[i], c) + α / K
 *
 *    loss = - (1/N) Σ_i Σ_c  smoothed_target[i][c] * log_softmax(logits[i][c])
 *
 *  Backward:
 *    grad[i][c] = (1/N) * (softmax(logits[i])[c] - smoothed_target[i][c])
 * ─────────────────────────────────────────────────────────────────── */

int label_smoothing_ce_forward(const double *logits,
                               const int64_t *targets,
                               int64_t batch_size,
                               int64_t num_classes,
                               double  smoothing,
                               double *loss_out)
{
    if (!logits || !targets || !loss_out)
        return -1;
    if (batch_size <= 0 || num_classes <= 0)
        return -1;
    if (smoothing < 0.0 || smoothing > 1.0)
        return -1;

    double total_loss = 0.0;

    for (int64_t i = 0; i < batch_size; i++) {
        const double *row = logits + i * num_classes;

        /* log-softmax: first find max for numerical stability */
        double max_val = row[0];
        for (int64_t c = 1; c < num_classes; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        /* log(sum(exp(x - max))) */
        double log_sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; c++) {
            log_sum_exp += exp(row[c] - max_val);
        }
        log_sum_exp = log(log_sum_exp);

        /* accumulate: -Σ_c smoothed_target[c] * log_softmax[c] */
        int64_t tgt = targets[i];
        if (tgt < 0 || tgt >= num_classes) return -1;

        double sample_loss = 0.0;
        for (int64_t c = 0; c < num_classes; c++) {
            double log_soft = (row[c] - max_val) - log_sum_exp;
            double smooth_t = smoothing / (double)num_classes;
            if (c == tgt)
                smooth_t += (1.0 - smoothing);
            sample_loss -= smooth_t * log_soft;
        }
        total_loss += sample_loss;
    }

    *loss_out = total_loss / (double)batch_size;
    return 0;
}


int label_smoothing_ce_backward(const double *logits,
                                const int64_t *targets,
                                int64_t batch_size,
                                int64_t num_classes,
                                double  smoothing,
                                double *grad_out)
{
    if (!logits || !targets || !grad_out)
        return -1;
    if (batch_size <= 0 || num_classes <= 0)
        return -1;
    if (smoothing < 0.0 || smoothing > 1.0)
        return -1;

    double inv_n = 1.0 / (double)batch_size;

    for (int64_t i = 0; i < batch_size; i++) {
        const double *row = logits + i * num_classes;
        double *grd = grad_out + i * num_classes;

        /* softmax */
        double max_val = row[0];
        for (int64_t c = 1; c < num_classes; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        double sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; c++) {
            grd[c] = exp(row[c] - max_val);
            sum_exp += grd[c];
        }
        /* grd[c] = softmax[c] */
        for (int64_t c = 0; c < num_classes; c++) {
            grd[c] /= sum_exp;
        }

        /* grad = inv_n * (softmax - smoothed_target) */
        int64_t tgt = targets[i];
        if (tgt < 0 || tgt >= num_classes) return -1;

        for (int64_t c = 0; c < num_classes; c++) {
            double smooth_t = smoothing / (double)num_classes;
            if (c == tgt)
                smooth_t += (1.0 - smoothing);
            grd[c] = inv_n * (grd[c] - smooth_t);
        }
    }
    return 0;
}


/* ───────────────────────────────────────────────────────────────────
 *  ROC-AUC Score  (binary classification)
 *
 *  Implements the trapezoidal-rule AUC computation:
 *    1. Sort samples by predicted score (descending).
 *    2. Walk through sorted list, accumulating TP / FP counts.
 *    3. Compute area under (FPR, TPR) curve via trapezoidal rule.
 *
 *  Returns -1 on error, 0 on success (AUC written to *auc_out).
 *  Edge cases:
 *    - All-positive or all-negative labels → AUC = 0.0 (undefined).
 * ─────────────────────────────────────────────────────────────────── */

/* Helper: index sort (descending by score, stable via index tie-break) */
typedef struct {
    double score;
    int64_t label;
    int64_t orig_idx;
} roc_sample_t;

static int roc_cmp_desc(const void *a, const void *b)
{
    const roc_sample_t *sa = (const roc_sample_t *)a;
    const roc_sample_t *sb = (const roc_sample_t *)b;
    if (sa->score > sb->score) return -1;
    if (sa->score < sb->score) return  1;
    /* tie-break by original index (ascending for stability) */
    if (sa->orig_idx < sb->orig_idx) return -1;
    if (sa->orig_idx > sb->orig_idx) return  1;
    return 0;
}

int roc_auc_score(const double *y_true,
                  const double *y_score,
                  int64_t n,
                  double *auc_out)
{
    if (!y_true || !y_score || !auc_out)
        return -1;
    if (n <= 0)
        return -1;

    /* count positives / negatives */
    int64_t num_pos = 0, num_neg = 0;
    for (int64_t i = 0; i < n; i++) {
        if (y_true[i] > 0.5) num_pos++;
        else                  num_neg++;
    }
    if (num_pos == 0 || num_neg == 0) {
        /* AUC undefined when only one class present */
        *auc_out = 0.0;
        return 0;
    }

    /* build sortable array */
    roc_sample_t *samples = (roc_sample_t *)malloc((size_t)n * sizeof(roc_sample_t));
    if (!samples) return -1;

    for (int64_t i = 0; i < n; i++) {
        samples[i].score    = y_score[i];
        samples[i].label    = (y_true[i] > 0.5) ? 1 : 0;
        samples[i].orig_idx = i;
    }
    qsort(samples, (size_t)n, sizeof(roc_sample_t), roc_cmp_desc);

    /* Trapezoidal AUC
     * Walk sorted samples, track (FPR, TPR) at each unique threshold.
     */
    double auc     = 0.0;
    double tp      = 0.0;
    double fp      = 0.0;
    double prev_tp = 0.0;
    double prev_fp = 0.0;

    for (int64_t i = 0; i < n; i++) {
        if (samples[i].label == 1)
            tp += 1.0;
        else
            fp += 1.0;

        /* At boundary: next sample has a different score or is the last one */
        if (i == n - 1 || samples[i].score != samples[i + 1].score) {
            double tpr      = tp / (double)num_pos;
            double fpr      = fp / (double)num_neg;
            double prev_tpr = prev_tp / (double)num_pos;
            double prev_fpr = prev_fp / (double)num_neg;

            /* trapezoidal area for this segment */
            auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr);

            prev_tp = tp;
            prev_fp = fp;
        }
    }

    free(samples);
    *auc_out = auc;
    return 0;
}
