/*
 * activations.c — Tensor-level activation functions (forward only).
 *
 * The assembly activations in activations.asm work at the autograd Node
 * level, which is great for the computation graph but doesn't match the
 * simple (out, input) tensor API that Python expects.  This file provides
 * thin C implementations that operate directly on NeuralTensor structs.
 *
 * Each function:
 *   1. Reads numel + data pointer + dtype from the input tensor
 *   2. Applies the activation element-wise
 *   3. Writes the result into the output tensor
 *
 * Both float32 and float64 are supported.  The math is done in double
 * precision either way — for float32 we just convert on the way in/out.
 *
 * Data layout of NeuralTensor (from tensor.asm):
 *   offset 0:  double *data
 *   offset 8:  int64_t ndim
 *   offset 16: int64_t *shape
 *   offset 24: int64_t *stride
 *   offset 32: int32_t dtype   (0 = float32, 1 = float64)
 *   offset 36: int32_t flags
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* Error codes — must match neural_api.h / neural_api.asm */
#define NEURAL_OK               0
#define NEURAL_ERR_NULL_POINTER  1
#define NEURAL_ERR_SHAPE_MISMATCH 4

/* Tensor struct matching the assembly layout */
typedef struct {
    void    *data;      /* offset 0  */
    int64_t  ndim;      /* offset 8  */
    int64_t *shape;     /* offset 16 */
    int64_t *stride;    /* offset 24 */
    int32_t  dtype;     /* offset 32, 0=f32, 1=f64 */
    int32_t  flags;     /* offset 36 */
} NeuralTensorC;

/* Total number of elements */
static int64_t tensor_numel_c(const NeuralTensorC *t)
{
    if (!t || t->ndim <= 0 || !t->shape) return 0;
    int64_t n = 1;
    for (int64_t i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}


/* ── Macros to reduce boilerplate ────────────────────────────────── */

/*
 * DEFINE_ACTIVATION_SIMPLE(name, expr)
 * Generates a function: int act_name(void *out, const void *in)
 * that applies `expr` element-wise.  `x` is the input value (double).
 */
#define DEFINE_ACTIVATION_SIMPLE(name, expr)                               \
int act_##name(void *out_ptr, const void *in_ptr)                          \
{                                                                          \
    if (!out_ptr || !in_ptr) return NEURAL_ERR_NULL_POINTER;               \
    NeuralTensorC *out = (NeuralTensorC *)out_ptr;                         \
    const NeuralTensorC *in = (const NeuralTensorC *)in_ptr;               \
    int64_t n = tensor_numel_c(in);                                        \
    if (n <= 0) return NEURAL_OK;                                          \
    if (in->dtype == 0) {  /* float32 */                                   \
        const float *src = (const float *)in->data;                        \
        float *dst = (float *)out->data;                                   \
        for (int64_t i = 0; i < n; i++) {                                  \
            double x = (double)src[i];                                     \
            dst[i] = (float)(expr);                                        \
        }                                                                  \
    } else {  /* float64 */                                                \
        const double *src = (const double *)in->data;                      \
        double *dst = (double *)out->data;                                 \
        for (int64_t i = 0; i < n; i++) {                                  \
            double x = src[i];                                             \
            dst[i] = (expr);                                               \
        }                                                                  \
    }                                                                      \
    return NEURAL_OK;                                                      \
}

/*
 * For activations that take an extra double parameter (e.g. alpha).
 */
#define DEFINE_ACTIVATION_PARAM(name, expr)                                 \
int act_##name(void *out_ptr, const void *in_ptr, double param)            \
{                                                                          \
    if (!out_ptr || !in_ptr) return NEURAL_ERR_NULL_POINTER;               \
    NeuralTensorC *out = (NeuralTensorC *)out_ptr;                         \
    const NeuralTensorC *in = (const NeuralTensorC *)in_ptr;               \
    int64_t n = tensor_numel_c(in);                                        \
    if (n <= 0) return NEURAL_OK;                                          \
    double alpha = param;                                                  \
    (void)alpha;                                                           \
    if (in->dtype == 0) {                                                  \
        const float *src = (const float *)in->data;                        \
        float *dst = (float *)out->data;                                   \
        for (int64_t i = 0; i < n; i++) {                                  \
            double x = (double)src[i];                                     \
            dst[i] = (float)(expr);                                        \
        }                                                                  \
    } else {                                                               \
        const double *src = (const double *)in->data;                      \
        double *dst = (double *)out->data;                                 \
        for (int64_t i = 0; i < n; i++) {                                  \
            double x = src[i];                                             \
            dst[i] = (expr);                                               \
        }                                                                  \
    }                                                                      \
    return NEURAL_OK;                                                      \
}


/* ── Activation implementations ──────────────────────────────────── */

/* tanh(x) — the real one, not sigmoid */
DEFINE_ACTIVATION_SIMPLE(tanh, tanh(x))

/* GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x*x*x))) */
static const double GELU_COEFF = 0.7978845608028654;   /* sqrt(2/pi) */
DEFINE_ACTIVATION_SIMPLE(gelu,
    0.5 * x * (1.0 + tanh(GELU_COEFF * (x + 0.044715 * x * x * x))))

/* LeakyReLU: max(alpha*x, x)  — alpha passed as param */
DEFINE_ACTIVATION_PARAM(leaky_relu, x >= 0 ? x : alpha * x)

/* ELU: x if x > 0, else alpha * (exp(x) - 1) */
DEFINE_ACTIVATION_PARAM(elu, x > 0 ? x : alpha * (exp(x) - 1.0))

/* SELU: lambda * (x if x > 0, else alpha * (exp(x) - 1)) */
static const double SELU_LAMBDA = 1.0507009873554804934193349852946;
static const double SELU_ALPHA  = 1.6732632423543772848170429916717;
DEFINE_ACTIVATION_SIMPLE(selu,
    x > 0 ? SELU_LAMBDA * x : SELU_LAMBDA * SELU_ALPHA * (exp(x) - 1.0))

/* Swish / SiLU: x * sigmoid(x) = x / (1 + exp(-x)) */
DEFINE_ACTIVATION_SIMPLE(swish, x / (1.0 + exp(-x)))

/* Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))) */
DEFINE_ACTIVATION_SIMPLE(mish, x * tanh(log(1.0 + exp(x))))

/* HardSwish: 0 if x<=-3, x if x>=3, else x*(x+3)/6 */
DEFINE_ACTIVATION_SIMPLE(hardswish,
    x <= -3.0 ? 0.0 : (x >= 3.0 ? x : x * (x + 3.0) / 6.0))

/* Softplus: ln(1 + exp(x)) */
DEFINE_ACTIVATION_SIMPLE(softplus, log(1.0 + exp(x)))

/* HardTanh: clamp to [-1, 1] */
DEFINE_ACTIVATION_SIMPLE(hardtanh,
    x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x))
