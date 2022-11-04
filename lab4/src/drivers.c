#include "drivers.h"
#include "blas.h"
#include "utils.h"

#include <stdio.h>
#include <string.h>

#define REPS 1000

stats_t* driver_cgs(size_t n, matrix_t* A, vector_t* x, size_t deg_m, size_t reps)
{
    stats_t* stats = stats_init("cgs", n);
    if (!stats) return NULL;

    matrix_t* Q = matrix_zeroes(A->rows, deg_m + 1);
    matrix_t* H = matrix_zeroes(deg_m + 1, deg_m);

    double elapsed;
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < reps; ++_) {
                classical_gram_schmidt(n, x->data, A->data, deg_m, Q, H);
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, reps);
            if (i == 0) {
                stats->resQ = dnrmf(Q->rows, Q->cols, Q->data);
                stats->resH = dnrmf(H->rows, H->cols, H->data);
            }
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_mgs(size_t n, matrix_t* A, vector_t* x, size_t deg_m, size_t reps)
{
    stats_t* stats = stats_init("mgs", n);
    if (!stats) return NULL;

    matrix_t* Q = matrix_zeroes(A->rows, deg_m + 1);
    matrix_t* H = matrix_zeroes(deg_m + 1, deg_m);

    double elapsed;
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < reps; ++_) {
                modified_gram_schmidt(n, x->data, A->data, deg_m, Q, H);
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, reps);
            if (i == 0) {
                stats->resQ = dnrmf(Q->rows, Q->cols, Q->data);
                stats->resH = dnrmf(H->rows, H->cols, H->data);
            }
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}
