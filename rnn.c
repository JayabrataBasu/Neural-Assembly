/*
 * rnn.c — LSTM and GRU recurrent layers (forward only).
 *
 * Part of the Neural-Assembly framework.
 *
 * Both layers operate on float64 data in the shape convention:
 *   input:  [batch, seq_len, input_size]    or [seq_len, input_size] if batch=1
 *   output: [batch, seq_len, hidden_size]   (full sequence)
 *   h_n:    [batch, hidden_size]            (last hidden state)
 *
 * Weight layout (column-major dense):
 *   LSTM has four gates (i, f, g, o), each with input and hidden weights
 *   plus a bias vector.  We store them concatenated:
 *     W_ih: [4*hidden, input_size]   — input weights for all four gates
 *     W_hh: [4*hidden, hidden_size]  — hidden weights for all four gates
 *     b_ih: [4*hidden]               — input bias
 *     b_hh: [4*hidden]               — hidden bias
 *
 *   GRU has three gates (r, z, n):
 *     W_ih: [3*hidden, input_size]
 *     W_hh: [3*hidden, hidden_size]
 *     b_ih: [3*hidden]
 *     b_hh: [3*hidden]
 *
 * Initialization: Xavier uniform for weights, zero for biases,
 * except the LSTM forget-gate bias which is set to 1.0 (Jozefowicz et al.)
 *
 * Functions:
 *   lstm_create    — allocate an LSTM layer
 *   lstm_free      — deallocate
 *   lstm_forward   — run the full sequence through the LSTM
 *   lstm_weight_ih — pointer to W_ih buffer
 *   lstm_weight_hh — pointer to W_hh buffer
 *   lstm_bias_ih   — pointer to b_ih buffer
 *   lstm_bias_hh   — pointer to b_hh buffer
 *
 *   gru_create     — allocate a GRU layer
 *   gru_free       — deallocate
 *   gru_forward    — run the full sequence through the GRU
 *   gru_weight_ih  — pointer to W_ih buffer
 *   gru_weight_hh  — pointer to W_hh buffer
 *   gru_bias_ih    — pointer to b_ih buffer
 *   gru_bias_hh    — pointer to b_hh buffer
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NEURAL_OK                0
#define NEURAL_ERR_NULL_POINTER  1
#define NEURAL_ERR_OUT_OF_MEMORY 2


/* ── Activations ─────────────────────────────────────────────────── */

static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


/* ── Xavier uniform init ─────────────────────────────────────────── */

/*
 * Simple xorshift64 PRNG — good enough for weight init.
 * We don't need cryptographic quality, just reasonable spread.
 */
static uint64_t rng_state = 0x12345678DEADBEEFULL;

static double rand_uniform(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFF) / 4294967296.0;
}

static void xavier_uniform_init(double *w, int64_t fan_in, int64_t fan_out,
                                int64_t n)
{
    double limit = sqrt(6.0 / (double)(fan_in + fan_out));
    for (int64_t i = 0; i < n; i++)
        w[i] = (2.0 * rand_uniform() - 1.0) * limit;
}


/* ── LSTM ─────────────────────────────────────────────────────────── */

typedef struct {
    int64_t input_size;
    int64_t hidden_size;
    double *W_ih;       /* [4*hidden, input_size]  */
    double *W_hh;       /* [4*hidden, hidden_size] */
    double *b_ih;       /* [4*hidden]              */
    double *b_hh;       /* [4*hidden]              */
} LSTMLayer;


void *lstm_create(int64_t input_size, int64_t hidden_size)
{
    LSTMLayer *l = calloc(1, sizeof(LSTMLayer));
    if (!l) return NULL;

    l->input_size  = input_size;
    l->hidden_size = hidden_size;
    int64_t h4 = 4 * hidden_size;

    l->W_ih = calloc(h4 * input_size,  sizeof(double));
    l->W_hh = calloc(h4 * hidden_size, sizeof(double));
    l->b_ih = calloc(h4, sizeof(double));
    l->b_hh = calloc(h4, sizeof(double));

    if (!l->W_ih || !l->W_hh || !l->b_ih || !l->b_hh) {
        free(l->W_ih); free(l->W_hh); free(l->b_ih); free(l->b_hh);
        free(l);
        return NULL;
    }

    /* Xavier init for weights */
    xavier_uniform_init(l->W_ih, input_size, hidden_size, h4 * input_size);
    xavier_uniform_init(l->W_hh, hidden_size, hidden_size, h4 * hidden_size);

    /* Forget gate bias = 1 (gate index 1 in the i,f,g,o ordering) */
    for (int64_t i = 0; i < hidden_size; i++) {
        l->b_ih[hidden_size + i] = 1.0;
    }

    return l;
}

void lstm_free(void *ptr)
{
    if (!ptr) return;
    LSTMLayer *l = (LSTMLayer *)ptr;
    free(l->W_ih);
    free(l->W_hh);
    free(l->b_ih);
    free(l->b_hh);
    free(l);
}

/*
 * lstm_forward — run one batch through the LSTM for all time steps.
 *
 *   input:   [batch, seq_len, input_size]    (row-major doubles)
 *   output:  [batch, seq_len, hidden_size]   (pre-allocated by caller)
 *   h_out:   [batch, hidden_size]            (last hidden, pre-allocated)
 *   c_out:   [batch, hidden_size]            (last cell, pre-allocated)
 *   h_init:  [batch, hidden_size] or NULL    (initial hidden state)
 *   c_init:  [batch, hidden_size] or NULL    (initial cell state)
 *   batch:   batch size
 *   seq_len: sequence length
 *
 * Returns 0 on success.
 */
int lstm_forward(void *layer_ptr,
                 const double *input, double *output,
                 double *h_out, double *c_out,
                 const double *h_init, const double *c_init,
                 int64_t batch, int64_t seq_len)
{
    if (!layer_ptr || !input || !output) return NEURAL_ERR_NULL_POINTER;

    LSTMLayer *l = (LSTMLayer *)layer_ptr;
    int64_t H = l->hidden_size;
    int64_t I = l->input_size;
    int64_t h4 = 4 * H;

    /* scratch: gates + h_prev + c_prev for one batch element */
    double *gates  = malloc(h4 * sizeof(double));
    double *h_prev = calloc(H, sizeof(double));
    double *c_prev = calloc(H, sizeof(double));
    if (!gates || !h_prev || !c_prev) {
        free(gates); free(h_prev); free(c_prev);
        return NEURAL_ERR_OUT_OF_MEMORY;
    }

    for (int64_t b = 0; b < batch; b++) {
        /* initialise h and c for this batch element */
        if (h_init)
            memcpy(h_prev, h_init + b * H, H * sizeof(double));
        else
            memset(h_prev, 0, H * sizeof(double));

        if (c_init)
            memcpy(c_prev, c_init + b * H, H * sizeof(double));
        else
            memset(c_prev, 0, H * sizeof(double));

        for (int64_t t = 0; t < seq_len; t++) {
            const double *x_t = input + b * seq_len * I + t * I;

            /* gates = W_ih @ x_t + b_ih + W_hh @ h_prev + b_hh */
            for (int64_t g = 0; g < h4; g++) {
                double sum = l->b_ih[g] + l->b_hh[g];

                /* W_ih[g, :] dot x_t */
                const double *w_row = l->W_ih + g * I;
                for (int64_t k = 0; k < I; k++)
                    sum += w_row[k] * x_t[k];

                /* W_hh[g, :] dot h_prev */
                const double *wh_row = l->W_hh + g * H;
                for (int64_t k = 0; k < H; k++)
                    sum += wh_row[k] * h_prev[k];

                gates[g] = sum;
            }

            /* split into i, f, g, o and apply activations */
            for (int64_t j = 0; j < H; j++) {
                double i_gate = sigmoid(gates[j]);              /* input gate  */
                double f_gate = sigmoid(gates[H + j]);          /* forget gate */
                double g_gate = tanh(gates[2 * H + j]);         /* cell gate   */
                double o_gate = sigmoid(gates[3 * H + j]);      /* output gate */

                c_prev[j] = f_gate * c_prev[j] + i_gate * g_gate;
                h_prev[j] = o_gate * tanh(c_prev[j]);
            }

            /* write h_t to output */
            double *out_t = output + b * seq_len * H + t * H;
            memcpy(out_t, h_prev, H * sizeof(double));
        }

        /* save final hidden and cell state */
        if (h_out) memcpy(h_out + b * H, h_prev, H * sizeof(double));
        if (c_out) memcpy(c_out + b * H, c_prev, H * sizeof(double));
    }

    free(gates);
    free(h_prev);
    free(c_prev);
    return NEURAL_OK;
}


/* ── GRU ──────────────────────────────────────────────────────────── */

typedef struct {
    int64_t input_size;
    int64_t hidden_size;
    double *W_ih;       /* [3*hidden, input_size]  */
    double *W_hh;       /* [3*hidden, hidden_size] */
    double *b_ih;       /* [3*hidden]              */
    double *b_hh;       /* [3*hidden]              */
} GRULayer;


void *gru_create(int64_t input_size, int64_t hidden_size)
{
    GRULayer *l = calloc(1, sizeof(GRULayer));
    if (!l) return NULL;

    l->input_size  = input_size;
    l->hidden_size = hidden_size;
    int64_t h3 = 3 * hidden_size;

    l->W_ih = calloc(h3 * input_size,  sizeof(double));
    l->W_hh = calloc(h3 * hidden_size, sizeof(double));
    l->b_ih = calloc(h3, sizeof(double));
    l->b_hh = calloc(h3, sizeof(double));

    if (!l->W_ih || !l->W_hh || !l->b_ih || !l->b_hh) {
        free(l->W_ih); free(l->W_hh); free(l->b_ih); free(l->b_hh);
        free(l);
        return NULL;
    }

    xavier_uniform_init(l->W_ih, input_size, hidden_size, h3 * input_size);
    xavier_uniform_init(l->W_hh, hidden_size, hidden_size, h3 * hidden_size);

    return l;
}

void gru_free(void *ptr)
{
    if (!ptr) return;
    GRULayer *l = (GRULayer *)ptr;
    free(l->W_ih);
    free(l->W_hh);
    free(l->b_ih);
    free(l->b_hh);
    free(l);
}

/*
 * gru_forward — run one batch through the GRU for all time steps.
 *
 * Same shape conventions as lstm_forward, minus the cell state.
 *
 * GRU equations:
 *   r = σ(W_ir @ x + b_ir + W_hr @ h + b_hr)       reset gate
 *   z = σ(W_iz @ x + b_iz + W_hz @ h + b_hz)       update gate
 *   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))   new gate
 *   h' = (1 - z) * n + z * h
 */
int gru_forward(void *layer_ptr,
                const double *input, double *output,
                double *h_out,
                const double *h_init,
                int64_t batch, int64_t seq_len)
{
    if (!layer_ptr || !input || !output) return NEURAL_ERR_NULL_POINTER;

    GRULayer *l = (GRULayer *)layer_ptr;
    int64_t H = l->hidden_size;
    int64_t I = l->input_size;
    int64_t h3 = 3 * H;

    double *gates_ih = malloc(h3 * sizeof(double));
    double *gates_hh = malloc(h3 * sizeof(double));
    double *h_prev   = calloc(H, sizeof(double));
    if (!gates_ih || !gates_hh || !h_prev) {
        free(gates_ih); free(gates_hh); free(h_prev);
        return NEURAL_ERR_OUT_OF_MEMORY;
    }

    for (int64_t b = 0; b < batch; b++) {
        if (h_init)
            memcpy(h_prev, h_init + b * H, H * sizeof(double));
        else
            memset(h_prev, 0, H * sizeof(double));

        for (int64_t t = 0; t < seq_len; t++) {
            const double *x_t = input + b * seq_len * I + t * I;

            /* gates_ih = W_ih @ x_t + b_ih */
            for (int64_t g = 0; g < h3; g++) {
                double sum = l->b_ih[g];
                const double *w_row = l->W_ih + g * I;
                for (int64_t k = 0; k < I; k++)
                    sum += w_row[k] * x_t[k];
                gates_ih[g] = sum;
            }

            /* gates_hh = W_hh @ h_prev + b_hh */
            for (int64_t g = 0; g < h3; g++) {
                double sum = l->b_hh[g];
                const double *w_row = l->W_hh + g * H;
                for (int64_t k = 0; k < H; k++)
                    sum += w_row[k] * h_prev[k];
                gates_hh[g] = sum;
            }

            for (int64_t j = 0; j < H; j++) {
                double r = sigmoid(gates_ih[j]     + gates_hh[j]);        /* reset  */
                double z = sigmoid(gates_ih[H + j] + gates_hh[H + j]);   /* update */
                double n = tanh(gates_ih[2*H + j] + r * gates_hh[2*H + j]); /* new */
                h_prev[j] = (1.0 - z) * n + z * h_prev[j];
            }

            double *out_t = output + b * seq_len * H + t * H;
            memcpy(out_t, h_prev, H * sizeof(double));
        }

        if (h_out) memcpy(h_out + b * H, h_prev, H * sizeof(double));
    }

    free(gates_ih);
    free(gates_hh);
    free(h_prev);
    return NEURAL_OK;
}


/* ── Weight accessors ─────────────────────────────────────────────── */

double *lstm_weight_ih(void *ptr) { return ptr ? ((LSTMLayer *)ptr)->W_ih : NULL; }
double *lstm_weight_hh(void *ptr) { return ptr ? ((LSTMLayer *)ptr)->W_hh : NULL; }
double *lstm_bias_ih(void *ptr)   { return ptr ? ((LSTMLayer *)ptr)->b_ih : NULL; }
double *lstm_bias_hh(void *ptr)   { return ptr ? ((LSTMLayer *)ptr)->b_hh : NULL; }

int64_t lstm_input_size(void *ptr)  { return ptr ? ((LSTMLayer *)ptr)->input_size  : 0; }
int64_t lstm_hidden_size(void *ptr) { return ptr ? ((LSTMLayer *)ptr)->hidden_size : 0; }

double *gru_weight_ih(void *ptr) { return ptr ? ((GRULayer *)ptr)->W_ih : NULL; }
double *gru_weight_hh(void *ptr) { return ptr ? ((GRULayer *)ptr)->W_hh : NULL; }
double *gru_bias_ih(void *ptr)   { return ptr ? ((GRULayer *)ptr)->b_ih : NULL; }
double *gru_bias_hh(void *ptr)   { return ptr ? ((GRULayer *)ptr)->b_hh : NULL; }

int64_t gru_input_size(void *ptr)  { return ptr ? ((GRULayer *)ptr)->input_size  : 0; }
int64_t gru_hidden_size(void *ptr) { return ptr ? ((GRULayer *)ptr)->hidden_size : 0; }
