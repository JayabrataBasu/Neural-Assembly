/* Dataset Prefetching Module
 * Async batch loading with pthread-based background threads
 * Decouples I/O latency from GPU/compute scheduling
 */

#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

/* Ring buffer for batches (thread-safe, lock-free) */
typedef struct {
    float **batch_data;      /* batch_count × max_batch_size */
    int *batch_labels;       /* batch_count entries */
    int *batch_sizes;        /* actual size of each batch */
    int batch_count;         /* capacity (ring size) */
    int write_index;         /* next write position */
    int read_index;          /* next read position */
    int pending;             /* batches waiting to be consumed */
    
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} PrefetchBuffer;

/* Prefetcher thread control */
typedef struct {
    PrefetchBuffer *buffer;
    void (*load_fn)(int, float *, int *, int);  /* dataset-specific loader */
    int num_batches;
    int batch_size;
    int *shuffle_idx;
    volatile int stop_flag;
    pthread_t thread;
    int current_epoch;
} DataPrefetcher;

/* ============================================================================
   Ring Buffer API
   ============================================================================ */

PrefetchBuffer *prefetch_buffer_create(int buffer_size, int max_batch_size) {
    PrefetchBuffer *pb = (PrefetchBuffer *)malloc(sizeof(PrefetchBuffer));
    pb->batch_count = buffer_size;
    pb->write_index = 0;
    pb->read_index = 0;
    pb->pending = 0;
    
    /* Allocate batch pointers and metadata */
    pb->batch_data = (float **)malloc(buffer_size * sizeof(float *));
    pb->batch_labels = (int *)malloc(buffer_size * sizeof(int));
    pb->batch_sizes = (int *)malloc(buffer_size * sizeof(int));
    
    for (int i = 0; i < buffer_size; i++) {
        pb->batch_data[i] = (float *)malloc(max_batch_size * sizeof(float));
        pb->batch_sizes[i] = 0;
    }
    
    pthread_mutex_init(&pb->lock, NULL);
    pthread_cond_init(&pb->not_empty, NULL);
    pthread_cond_init(&pb->not_full, NULL);
    
    return pb;
}

void prefetch_buffer_free(PrefetchBuffer *pb) {
    if (!pb) return;
    for (int i = 0; i < pb->batch_count; i++) {
        free(pb->batch_data[i]);
    }
    free(pb->batch_data);
    free(pb->batch_labels);
    free(pb->batch_sizes);
    pthread_mutex_destroy(&pb->lock);
    pthread_cond_destroy(&pb->not_empty);
    pthread_cond_destroy(&pb->not_full);
    free(pb);
}

/* Producer: Write batch to ring buffer (blocks if full) */
int prefetch_buffer_produce(PrefetchBuffer *pb, float *data, int label, int size) {
    pthread_mutex_lock(&pb->lock);
    
    while (pb->pending >= pb->batch_count - 1) {
        pthread_cond_wait(&pb->not_full, &pb->lock);
    }
    
    memcpy(pb->batch_data[pb->write_index], data, size * sizeof(float));
    pb->batch_labels[pb->write_index] = label;
    pb->batch_sizes[pb->write_index] = size;
    
    pb->write_index = (pb->write_index + 1) % pb->batch_count;
    pb->pending++;
    
    pthread_cond_signal(&pb->not_empty);
    pthread_mutex_unlock(&pb->lock);
    
    return 0;
}

/* Consumer: Read batch from ring buffer (blocks if empty) */
int prefetch_buffer_consume(PrefetchBuffer *pb, float **data, int *label, int *size) {
    pthread_mutex_lock(&pb->lock);
    
    while (pb->pending == 0) {
        pthread_cond_wait(&pb->not_empty, &pb->lock);
    }
    
    *data = pb->batch_data[pb->read_index];
    *label = pb->batch_labels[pb->read_index];
    *size = pb->batch_sizes[pb->read_index];
    
    pb->read_index = (pb->read_index + 1) % pb->batch_count;
    pb->pending--;
    
    pthread_cond_signal(&pb->not_full);
    pthread_mutex_unlock(&pb->lock);
    
    return 0;
}

/* ============================================================================
   Prefetcher Thread
   ============================================================================ */

void *prefetch_worker_thread(void *arg) {
    DataPrefetcher *prefetcher = (DataPrefetcher *)arg;
    PrefetchBuffer *pb = prefetcher->buffer;
    int current_idx = 0;
    
    while (!prefetcher->stop_flag) {
        /* Load next batch via callback */
        float *batch = pb->batch_data[pb->write_index];
        int label = prefetcher->shuffle_idx[current_idx];
        int batch_idx = current_idx;
        
        prefetcher->load_fn(batch_idx, batch, &label, prefetcher->batch_size);
        
        /* Write to buffer */
        prefetch_buffer_produce(pb, batch, label, prefetcher->batch_size);
        
        current_idx++;
        if (current_idx >= prefetcher->num_batches) {
            current_idx = 0;
            prefetcher->current_epoch++;
        }
    }
    
    return NULL;
}

/* ============================================================================
   Public API
   ============================================================================ */

DataPrefetcher *dataprefetcher_create(
    int num_batches,
    int batch_size,
    int buffer_size,
    void (*load_fn)(int, float *, int *, int)
) {
    DataPrefetcher *prefetcher = (DataPrefetcher *)malloc(sizeof(DataPrefetcher));
    prefetcher->buffer = prefetch_buffer_create(buffer_size, batch_size);
    prefetcher->load_fn = load_fn;
    prefetcher->num_batches = num_batches;
    prefetcher->batch_size = batch_size;
    prefetcher->stop_flag = 0;
    prefetcher->current_epoch = 0;
    
    /* Shuffle indices */
    prefetcher->shuffle_idx = (int *)malloc(num_batches * sizeof(int));
    for (int i = 0; i < num_batches; i++) {
        prefetcher->shuffle_idx[i] = i;
    }
    
    return prefetcher;
}

void dataprefetcher_start(DataPrefetcher *prefetcher) {
    pthread_create(&prefetcher->thread, NULL, prefetch_worker_thread, prefetcher);
}

void dataprefetcher_stop(DataPrefetcher *prefetcher) {
    prefetcher->stop_flag = 1;
    pthread_join(prefetcher->thread, NULL);
}

void dataprefetcher_shuffle(DataPrefetcher *prefetcher) {
    /* Fisher-Yates shuffle */
    for (int i = prefetcher->num_batches - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = prefetcher->shuffle_idx[i];
        prefetcher->shuffle_idx[i] = prefetcher->shuffle_idx[j];
        prefetcher->shuffle_idx[j] = tmp;
    }
}

int dataprefetcher_next_batch(DataPrefetcher *prefetcher, float **batch, int *label, int *size) {
    return prefetch_buffer_consume(prefetcher->buffer, batch, label, size);
}

void dataprefetcher_free(DataPrefetcher *prefetcher) {
    if (!prefetcher) return;
    prefetch_buffer_free(prefetcher->buffer);
    free(prefetcher->shuffle_idx);
    free(prefetcher);
}

int dataprefetcher_pending_count(DataPrefetcher *prefetcher) {
    pthread_mutex_lock(&prefetcher->buffer->lock);
    int count = prefetcher->buffer->pending;
    pthread_mutex_unlock(&prefetcher->buffer->lock);
    return count;
}

int dataprefetcher_current_epoch(DataPrefetcher *prefetcher) {
    return prefetcher->current_epoch;
}
