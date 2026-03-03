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
#define NEURAL_ERR_INVALID_ARGUMENT 3

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

typedef struct {
    uint32_t magic;        /* 'OPTS' */
    uint32_t version;      /* 1 */
    int32_t  type;
    int32_t  reserved;
    int64_t  n_params;
    double   lr;
    double   momentum;
    double   beta1;
    double   beta2;
    double   epsilon;
    double   weight_decay;
    int64_t  t;
} OptStateHeader;

#define OPT_STATE_MAGIC   0x5354504Fu  /* 'OPTS' */
#define OPT_STATE_VERSION 1u

static void free_moments(OptC *o)
{
    if (!o) return;
    for (int64_t i = 0; i < o->n_params; i++) {
        if (o->m)          free(o->m[i]);
        if (o->v)          free(o->v[i]);
        if (o->velocities) free(o->velocities[i]);
    }
    free(o->m);
    free(o->v);
    free(o->velocities);
    free(o->param_sizes);
    o->m = NULL;
    o->v = NULL;
    o->velocities = NULL;
    o->param_sizes = NULL;
    o->n_params = 0;
}

static int64_t total_param_elems(const OptC *o)
{
    if (!o || o->n_params <= 0 || !o->param_sizes) return 0;
    int64_t total = 0;
    for (int64_t i = 0; i < o->n_params; i++) {
        if (o->param_sizes[i] < 0) return -1;
        total += o->param_sizes[i];
    }
    return total;
}


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

    if (n < 0) return NEURAL_ERR_INVALID_ARGUMENT;
    if (o->n_params > 0 && n != o->n_params) return NEURAL_ERR_INVALID_ARGUMENT;

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
    free_moments(o);
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

/*
 * State export format:
 * [OptStateHeader]
 * [param_sizes: n_params * int64_t]
 * [moments/velocities data as doubles]
 *   SGD   : velocities
 *   Adam* : m then v
 */
int64_t opt_state_bytes(void *opt)
{
    if (!opt) return 0;
    OptC *o = (OptC *)opt;

    int64_t total = sizeof(OptStateHeader);
    total += o->n_params * (int64_t)sizeof(int64_t);

    int64_t elems = total_param_elems(o);
    if (elems < 0) return 0;

    if (o->type == OPT_SGD) {
        total += elems * (int64_t)sizeof(double);
    } else if (o->type == OPT_ADAM || o->type == OPT_ADAMW) {
        total += 2 * elems * (int64_t)sizeof(double);
    }
    return total;
}

int opt_state_export(void *opt, void *buffer, int64_t size)
{
    if (!opt || !buffer || size <= 0) return NEURAL_ERR_NULL_POINTER;
    OptC *o = (OptC *)opt;

    int64_t need = opt_state_bytes(opt);
    if (need <= 0 || size < need) return NEURAL_ERR_INVALID_ARGUMENT;

    uint8_t *p = (uint8_t *)buffer;

    OptStateHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = OPT_STATE_MAGIC;
    h.version = OPT_STATE_VERSION;
    h.type = o->type;
    h.n_params = o->n_params;
    h.lr = o->lr;
    h.momentum = o->momentum;
    h.beta1 = o->beta1;
    h.beta2 = o->beta2;
    h.epsilon = o->epsilon;
    h.weight_decay = o->weight_decay;
    h.t = o->t;

    memcpy(p, &h, sizeof(h));
    p += sizeof(h);

    if (o->n_params > 0 && o->param_sizes) {
        memcpy(p, o->param_sizes, (size_t)o->n_params * sizeof(int64_t));
        p += (size_t)o->n_params * sizeof(int64_t);
    }

    for (int64_t i = 0; i < o->n_params; i++) {
        int64_t n = o->param_sizes ? o->param_sizes[i] : 0;
        if (n <= 0) continue;
        if (o->type == OPT_SGD && o->velocities && o->velocities[i]) {
            memcpy(p, o->velocities[i], (size_t)n * sizeof(double));
            p += (size_t)n * sizeof(double);
        }
    }

    if (o->type == OPT_ADAM || o->type == OPT_ADAMW) {
        for (int64_t i = 0; i < o->n_params; i++) {
            int64_t n = o->param_sizes ? o->param_sizes[i] : 0;
            if (n <= 0) continue;
            if (o->m && o->m[i]) {
                memcpy(p, o->m[i], (size_t)n * sizeof(double));
                p += (size_t)n * sizeof(double);
            }
        }
        for (int64_t i = 0; i < o->n_params; i++) {
            int64_t n = o->param_sizes ? o->param_sizes[i] : 0;
            if (n <= 0) continue;
            if (o->v && o->v[i]) {
                memcpy(p, o->v[i], (size_t)n * sizeof(double));
                p += (size_t)n * sizeof(double);
            }
        }
    }

    return NEURAL_OK;
}

int opt_state_import(void *opt, const void *buffer, int64_t size)
{
    if (!opt || !buffer || size <= 0) return NEURAL_ERR_NULL_POINTER;
    OptC *o = (OptC *)opt;

    if (size < (int64_t)sizeof(OptStateHeader)) return NEURAL_ERR_INVALID_ARGUMENT;

    const uint8_t *p = (const uint8_t *)buffer;
    OptStateHeader h;
    memcpy(&h, p, sizeof(h));
    p += sizeof(h);

    if (h.magic != OPT_STATE_MAGIC || h.version != OPT_STATE_VERSION) {
        return NEURAL_ERR_INVALID_ARGUMENT;
    }
    if (h.type != o->type || h.n_params < 0) return NEURAL_ERR_INVALID_ARGUMENT;

    free_moments(o);

    o->lr = h.lr;
    o->momentum = h.momentum;
    o->beta1 = h.beta1;
    o->beta2 = h.beta2;
    o->epsilon = h.epsilon;
    o->weight_decay = h.weight_decay;
    o->t = h.t;
    o->n_params = h.n_params;

    if (o->n_params == 0) return NEURAL_OK;

    int64_t sizes_bytes = o->n_params * (int64_t)sizeof(int64_t);
    if ((int64_t)(p - (const uint8_t *)buffer) + sizes_bytes > size) return NEURAL_ERR_INVALID_ARGUMENT;

    o->param_sizes = calloc((size_t)o->n_params, sizeof(int64_t));
    if (!o->param_sizes) return NEURAL_ERR_OUT_OF_MEMORY;
    memcpy(o->param_sizes, p, (size_t)sizes_bytes);
    p += sizes_bytes;

    if (o->type == OPT_SGD) {
        o->velocities = calloc((size_t)o->n_params, sizeof(double *));
        if (!o->velocities) return NEURAL_ERR_OUT_OF_MEMORY;
        for (int64_t i = 0; i < o->n_params; i++) {
            int64_t n = o->param_sizes[i];
            if (n < 0) return NEURAL_ERR_INVALID_ARGUMENT;
            if (n == 0) continue;
            int64_t bytes = n * (int64_t)sizeof(double);
            if ((int64_t)(p - (const uint8_t *)buffer) + bytes > size) return NEURAL_ERR_INVALID_ARGUMENT;
            o->velocities[i] = calloc((size_t)n, sizeof(double));
            if (!o->velocities[i]) return NEURAL_ERR_OUT_OF_MEMORY;
            memcpy(o->velocities[i], p, (size_t)bytes);
            p += bytes;
        }
    } else if (o->type == OPT_ADAM || o->type == OPT_ADAMW) {
        o->m = calloc((size_t)o->n_params, sizeof(double *));
        o->v = calloc((size_t)o->n_params, sizeof(double *));
        if (!o->m || !o->v) return NEURAL_ERR_OUT_OF_MEMORY;

        for (int64_t i = 0; i < o->n_params; i++) {
            int64_t n = o->param_sizes[i];
            if (n < 0) return NEURAL_ERR_INVALID_ARGUMENT;
            if (n == 0) continue;
            int64_t bytes = n * (int64_t)sizeof(double);
            if ((int64_t)(p - (const uint8_t *)buffer) + bytes > size) return NEURAL_ERR_INVALID_ARGUMENT;
            o->m[i] = calloc((size_t)n, sizeof(double));
            if (!o->m[i]) return NEURAL_ERR_OUT_OF_MEMORY;
            memcpy(o->m[i], p, (size_t)bytes);
            p += bytes;
        }
        for (int64_t i = 0; i < o->n_params; i++) {
            int64_t n = o->param_sizes[i];
            if (n < 0) return NEURAL_ERR_INVALID_ARGUMENT;
            if (n == 0) continue;
            int64_t bytes = n * (int64_t)sizeof(double);
            if ((int64_t)(p - (const uint8_t *)buffer) + bytes > size) return NEURAL_ERR_INVALID_ARGUMENT;
            o->v[i] = calloc((size_t)n, sizeof(double));
            if (!o->v[i]) return NEURAL_ERR_OUT_OF_MEMORY;
            memcpy(o->v[i], p, (size_t)bytes);
            p += bytes;
        }
    }

    return NEURAL_OK;
}
