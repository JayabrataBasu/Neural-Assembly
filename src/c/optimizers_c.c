/*
 * optimizers_c.c — Python-friendly optimizer implementations.
 *
 * The assembly optimizers in optimizers.asm are built around the autograd
 * graph: they store param *nodes* at creation time and pull gradients from
 * node->grad during each step.  That's perfect for the internal training
 * loop but doesn't match the Python API, which passes separate param and
 * grad tensors on each step() call.
 *
 * This file provides lightweight C optimizers that follow the Python
 * calling convention:
 *
 *   void *opt = opt_adam_create(lr, beta1, beta2, eps);
 *   opt_step(opt, param_ptrs, grad_ptrs, n);
 *
 * Each optimizer allocates per-parameter moment buffers lazily (on the
 * first step call) so the user doesn't need to declare params up front.
 *
 * Tensor struct layout (from tensor.asm):
 *   offset 0:  void    *data
 *   offset 8:  int64_t  ndim
 *   offset 16: int64_t *shape
 *   offset 24: int64_t *stride
 *   offset 32: int32_t  dtype   (0 = f32, 1 = f64)
 *   offset 36: int32_t  flags
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Match error codes from neural_api.h */
#define NEURAL_OK               0
#define NEURAL_ERR_NULL_POINTER  1
#define NEURAL_ERR_OUT_OF_MEMORY 2

/* Tensor struct — same layout as tensor.asm */
typedef struct {
    void    *data;
    int64_t  ndim;
    int64_t *shape;
    int64_t *stride;
    int32_t  dtype;
    int32_t  flags;
} TensorC;

static int64_t numel(const TensorC *t)
{
    if (!t || t->ndim <= 0 || !t->shape) return 0;
    int64_t n = 1;
    for (int64_t i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* ── Optimizer types ──────────────────────────────────────────────── */

#define OPT_SGD   0
#define OPT_ADAM  1
#define OPT_ADAMW 2

typedef struct {
    int     type;
    double  lr;

    /* SGD */
    double  momentum;
    double **velocities;    /* per-param velocity arrays (alloc'd lazily) */

    /* Adam / AdamW */
    double  beta1, beta2, epsilon;
    double  weight_decay;   /* 0 for plain Adam */
    int64_t t;              /* time step */
    double **m;             /* first moment per-param */
    double **v;             /* second moment per-param */

    /* bookkeeping for lazy alloc */
    int64_t  n_params;      /* set on first step */
    int64_t *param_sizes;   /* numel for each param slot */
} OptC;


/* ── Creation ─────────────────────────────────────────────────────── */

void *opt_sgd_create(double lr, double momentum)
{
    OptC *o = calloc(1, sizeof(OptC));
    if (!o) return NULL;
    o->type = OPT_SGD;
    o->lr = lr;
    o->momentum = momentum;
    return o;
}

void *opt_adam_create(double lr, double beta1, double beta2, double epsilon)
{
    OptC *o = calloc(1, sizeof(OptC));
    if (!o) return NULL;
    o->type = OPT_ADAM;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->epsilon = epsilon;
    o->weight_decay = 0.0;
    return o;
}

void *opt_adamw_create(double lr, double beta1, double beta2,
                       double epsilon, double weight_decay)
{
    OptC *o = calloc(1, sizeof(OptC));
    if (!o) return NULL;
    o->type = OPT_ADAMW;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->epsilon = epsilon;
    o->weight_decay = weight_decay;
    return o;
}


/* ── Lazy moment allocation ───────────────────────────────────────── */

/*
 * Called on first step.  Allocates zero-filled moment vectors for each
 * parameter.  Returns 0 on success.
 */
static int alloc_moments_adam(OptC *o, void **params, int64_t n)
{
    o->n_params = n;
    o->param_sizes = calloc(n, sizeof(int64_t));
    o->m = calloc(n, sizeof(double *));
    o->v = calloc(n, sizeof(double *));
    if (!o->param_sizes || !o->m || !o->v) return -1;

    for (int64_t i = 0; i < n; i++) {
        TensorC *p = (TensorC *)params[i];
        int64_t sz = numel(p);
        o->param_sizes[i] = sz;
        o->m[i] = calloc(sz, sizeof(double));
        o->v[i] = calloc(sz, sizeof(double));
        if (!o->m[i] || !o->v[i]) return -1;
    }
    return 0;
}

static int alloc_velocities_sgd(OptC *o, void **params, int64_t n)
{
    o->n_params = n;
    o->param_sizes = calloc(n, sizeof(int64_t));
    o->velocities = calloc(n, sizeof(double *));
    if (!o->param_sizes || !o->velocities) return -1;

    for (int64_t i = 0; i < n; i++) {
        TensorC *p = (TensorC *)params[i];
        int64_t sz = numel(p);
        o->param_sizes[i] = sz;
        o->velocities[i] = calloc(sz, sizeof(double));
        if (!o->velocities[i]) return -1;
    }
    return 0;
}


/* ── Step implementations ─────────────────────────────────────────── */

static void sgd_step_impl(OptC *o, void **params, void **grads, int64_t n)
{
    /* lazy init */
    if (o->n_params == 0 && n > 0) {
        if (alloc_velocities_sgd(o, params, n) != 0) return;
    }

    for (int64_t i = 0; i < n; i++) {
        TensorC *p = (TensorC *)params[i];
        TensorC *g = (TensorC *)grads[i];
        int64_t sz = o->param_sizes[i];

        if (p->dtype == 0) {
            float *pd = (float *)p->data;
            float *gd = (float *)g->data;
            double *vel = o->velocities[i];
            for (int64_t j = 0; j < sz; j++) {
                double grad = (double)gd[j];
                vel[j] = o->momentum * vel[j] + grad;
                pd[j] -= (float)(o->lr * vel[j]);
            }
        } else {
            double *pd = (double *)p->data;
            double *gd = (double *)g->data;
            double *vel = o->velocities[i];
            for (int64_t j = 0; j < sz; j++) {
                vel[j] = o->momentum * vel[j] + gd[j];
                pd[j] -= o->lr * vel[j];
            }
        }
    }
}

static void adam_step_impl(OptC *o, void **params, void **grads, int64_t n)
{
    /* lazy init */
    if (o->n_params == 0 && n > 0) {
        if (alloc_moments_adam(o, params, n) != 0) return;
    }

    o->t++;
    double bc1 = 1.0 - pow(o->beta1, (double)o->t);
    double bc2 = 1.0 - pow(o->beta2, (double)o->t);

    for (int64_t i = 0; i < n; i++) {
        TensorC *p = (TensorC *)params[i];
        TensorC *g = (TensorC *)grads[i];
        int64_t sz = o->param_sizes[i];
        double *mi = o->m[i];
        double *vi = o->v[i];

        if (p->dtype == 0) {
            float *pd = (float *)p->data;
            float *gd = (float *)g->data;
            for (int64_t j = 0; j < sz; j++) {
                double grad = (double)gd[j];
                mi[j] = o->beta1 * mi[j] + (1.0 - o->beta1) * grad;
                vi[j] = o->beta2 * vi[j] + (1.0 - o->beta2) * grad * grad;
                double m_hat = mi[j] / bc1;
                double v_hat = vi[j] / bc2;
                pd[j] -= (float)(o->lr * m_hat / (sqrt(v_hat) + o->epsilon));
            }
        } else {
            double *pd = (double *)p->data;
            double *gd = (double *)g->data;
            for (int64_t j = 0; j < sz; j++) {
                double grad = gd[j];
                mi[j] = o->beta1 * mi[j] + (1.0 - o->beta1) * grad;
                vi[j] = o->beta2 * vi[j] + (1.0 - o->beta2) * grad * grad;
                double m_hat = mi[j] / bc1;
                double v_hat = vi[j] / bc2;
                pd[j] -= o->lr * m_hat / (sqrt(v_hat) + o->epsilon);
            }
        }
    }
}

static void adamw_step_impl(OptC *o, void **params, void **grads, int64_t n)
{
    /* lazy init */
    if (o->n_params == 0 && n > 0) {
        if (alloc_moments_adam(o, params, n) != 0) return;
    }

    o->t++;
    double bc1 = 1.0 - pow(o->beta1, (double)o->t);
    double bc2 = 1.0 - pow(o->beta2, (double)o->t);

    for (int64_t i = 0; i < n; i++) {
        TensorC *p = (TensorC *)params[i];
        TensorC *g = (TensorC *)grads[i];
        int64_t sz = o->param_sizes[i];
        double *mi = o->m[i];
        double *vi = o->v[i];

        if (p->dtype == 0) {
            float *pd = (float *)p->data;
            float *gd = (float *)g->data;
            for (int64_t j = 0; j < sz; j++) {
                double param = (double)pd[j];
                double grad  = (double)gd[j];

                /* decoupled weight decay: applied to param, not grad */
                param -= o->lr * o->weight_decay * param;

                mi[j] = o->beta1 * mi[j] + (1.0 - o->beta1) * grad;
                vi[j] = o->beta2 * vi[j] + (1.0 - o->beta2) * grad * grad;
                double m_hat = mi[j] / bc1;
                double v_hat = vi[j] / bc2;
                param -= o->lr * m_hat / (sqrt(v_hat) + o->epsilon);
                pd[j] = (float)param;
            }
        } else {
            double *pd = (double *)p->data;
            double *gd = (double *)g->data;
            for (int64_t j = 0; j < sz; j++) {
                double param = pd[j];
                double grad  = gd[j];

                param -= o->lr * o->weight_decay * param;

                mi[j] = o->beta1 * mi[j] + (1.0 - o->beta1) * grad;
                vi[j] = o->beta2 * vi[j] + (1.0 - o->beta2) * grad * grad;
                double m_hat = mi[j] / bc1;
                double v_hat = vi[j] / bc2;
                param -= o->lr * m_hat / (sqrt(v_hat) + o->epsilon);
                pd[j] = param;
            }
        }
    }
}


/* ── Public API ───────────────────────────────────────────────────── */

int opt_step(void *opt, void **params, void **grads, int64_t n)
{
    if (!opt || !params || !grads) return NEURAL_ERR_NULL_POINTER;
    OptC *o = (OptC *)opt;

    switch (o->type) {
        case OPT_SGD:   sgd_step_impl(o, params, grads, n);   break;
        case OPT_ADAM:  adam_step_impl(o, params, grads, n);   break;
        case OPT_ADAMW: adamw_step_impl(o, params, grads, n); break;
        default: return NEURAL_ERR_NULL_POINTER;
    }
    return NEURAL_OK;
}

void opt_free(void *opt)
{
    if (!opt) return;
    OptC *o = (OptC *)opt;

    for (int64_t i = 0; i < o->n_params; i++) {
        if (o->m)          free(o->m[i]);
        if (o->v)          free(o->v[i]);
        if (o->velocities) free(o->velocities[i]);
    }
    free(o->m);
    free(o->v);
    free(o->velocities);
    free(o->param_sizes);
    free(o);
}

double opt_get_lr(void *opt)
{
    if (!opt) return 0.0;
    return ((OptC *)opt)->lr;
}

void opt_set_lr(void *opt, double lr)
{
    if (!opt) return;
    ((OptC *)opt)->lr = lr;
}
