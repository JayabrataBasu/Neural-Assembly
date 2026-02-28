/*
 * embedding.c — Embedding lookup layer (forward + backward)
 *
 * Part of the Neural-Assembly framework.
 *
 * An Embedding stores a weight matrix W[num_embeddings × embedding_dim] (double).
 * Forward:  given int64 indices[seq_len], output[seq_len × embedding_dim] = W[indices[i], :]
 * Backward: accumulate gradients into grad_weight from grad_output.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct Embedding {
    int64_t num_embeddings;
    int64_t embedding_dim;
    double *weight;      /* [num_embeddings × embedding_dim] */
} Embedding;


Embedding *embedding_create(int64_t num_embeddings, int64_t embedding_dim)
{
    if (num_embeddings <= 0 || embedding_dim <= 0)
        return NULL;

    Embedding *emb = (Embedding *)calloc(1, sizeof(Embedding));
    if (!emb) return NULL;

    emb->num_embeddings = num_embeddings;
    emb->embedding_dim  = embedding_dim;
    emb->weight = (double *)calloc((size_t)(num_embeddings * embedding_dim), sizeof(double));
    if (!emb->weight) {
        free(emb);
        return NULL;
    }

    /* Xavier-uniform init: U(-sqrt(1/dim), sqrt(1/dim)) */
    double scale = sqrt(1.0 / (double)embedding_dim);
    for (int64_t i = 0; i < num_embeddings * embedding_dim; i++) {
        /* Simple LCG PRNG — good enough for initialisation */
        static uint64_t rng_state = 0x123456789ABCDEF0ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = ((double)(rng_state >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
        emb->weight[i] = u * scale;
    }

    return emb;
}


void embedding_free(Embedding *emb)
{
    if (emb) {
        free(emb->weight);
        free(emb);
    }
}


/* Forward: output[i * dim .. (i+1)*dim-1] = weight[indices[i] * dim .. ] */
int embedding_forward(const Embedding *emb,
                      const int64_t *indices,
                      int64_t seq_len,
                      double *output)
{
    if (!emb || !indices || !output)
        return -1;
    if (seq_len <= 0)
        return -1;

    int64_t dim = emb->embedding_dim;
    for (int64_t i = 0; i < seq_len; i++) {
        int64_t idx = indices[i];
        if (idx < 0 || idx >= emb->num_embeddings)
            return -1;
        memcpy(output + i * dim,
               emb->weight + idx * dim,
               (size_t)dim * sizeof(double));
    }
    return 0;
}


/* Backward: accumulate grad_output into grad_weight at the looked-up rows.
 * grad_weight must be zeroed before calling if you want a fresh gradient. */
int embedding_backward(const Embedding *emb,
                       const int64_t *indices,
                       int64_t seq_len,
                       const double *grad_output,
                       double *grad_weight)
{
    if (!emb || !indices || !grad_output || !grad_weight)
        return -1;
    if (seq_len <= 0)
        return -1;

    int64_t dim = emb->embedding_dim;
    for (int64_t i = 0; i < seq_len; i++) {
        int64_t idx = indices[i];
        if (idx < 0 || idx >= emb->num_embeddings)
            return -1;
        double *dst = grad_weight + idx * dim;
        const double *src = grad_output + i * dim;
        for (int64_t d = 0; d < dim; d++) {
            dst[d] += src[d];  /* accumulate */
        }
    }
    return 0;
}


/* Accessors */
double *embedding_weight(Embedding *emb)
{
    return emb ? emb->weight : NULL;
}

int64_t embedding_num_embeddings(Embedding *emb)
{
    return emb ? emb->num_embeddings : 0;
}

int64_t embedding_dim(Embedding *emb)
{
    return emb ? emb->embedding_dim : 0;
}
