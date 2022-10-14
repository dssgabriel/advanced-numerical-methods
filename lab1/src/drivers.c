#include "drivers.h"

#include "blas1.h"
#include "blas2.h"
#include "blas3.h"
#include "utils.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

#define REPS 1000

stats_t* driver_daxpy(config_t cfg, double a, vector_t* x, vector_t* y)
{
    stats_t* stats =
        stats_init("daxpy", 1, cfg.nb_threads, vector_nb_elems(x) + vector_nb_elems(y), 2);
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas1_daxpy(vector_nb_elems(y), a, x->data, y->data);
                }
                else {
                    blas1_daxpy(vector_nb_elems(y), a, x->data, y->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_ddot(config_t cfg, vector_t* x, vector_t* y)
{
    stats_t* stats =
        stats_init("ddot", 1, cfg.nb_threads, vector_nb_elems(x) + vector_nb_elems(y), 2);
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas1_ddot(vector_nb_elems(x), x->data, y->data);
                }
                else {
                    blas1_ddot(vector_nb_elems(x), x->data, y->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dnrm2(config_t cfg, vector_t* x)
{
    stats_t* stats = stats_init("dnrm2", 1, cfg.nb_threads, vector_nb_elems(x), 2);
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas1_dnrm2(vector_nb_elems(x), x->data);
                }
                else {
                    blas1_dnrm2(vector_nb_elems(x), x->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dmax(config_t cfg, vector_t* x)
{
    stats_t* stats = stats_init("dmax", 1, cfg.nb_threads, vector_nb_elems(x), 1);
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas1_dnrm2(vector_nb_elems(x), x->data);
                }
                else {
                    blas1_dnrm2(vector_nb_elems(x), x->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dgemv(config_t cfg, double alpha, matrix_t* A, vector_t* x, double beta,
                      vector_t* y)
{
    stats_t* stats = stats_init("dgemv", 2, cfg.nb_threads,
                                matrix_nb_elems(A) + vector_nb_elems(x) + vector_nb_elems(y),
                                3 * A->rows * (2 * A->cols));
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas2_dgemv(A->rows, A->cols, alpha, A->data, x->data, beta, y->data);
                }
                else {
                    blas2_dgemv(A->rows, A->cols, alpha, A->data, x->data, beta, y->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dgemv_var(config_t cfg, double alpha, matrix_t* A, vector_t* x, double beta,
                          vector_t* y)
{
    stats_t* stats = stats_init("dgemv_var", 2, cfg.nb_threads,
                                matrix_nb_elems(A) + vector_nb_elems(x) + vector_nb_elems(y),
                                3 * A->rows * (2 * A->cols));
    if (!stats)
        return NULL;

    matrix_t* AT = matrix_copy(A);
    if (!AT)
        return NULL;
    matrix_transpose(A, AT);

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas2_dgemv(AT->rows, AT->cols, alpha, AT->data, x->data, beta,
                                         y->data);
                }
                else {
                    blas2_dgemv(AT->rows, AT->cols, alpha, AT->data, x->data, beta, y->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    matrix_deinit(AT);
    stats_compute(stats);
    return stats;
}

stats_t* driver_dger(config_t cfg, double alpha, matrix_t* A, vector_t* x, vector_t* yT)
{
    stats_t* stats = stats_init("dger", 2, cfg.nb_threads,
                                matrix_nb_elems(A) + vector_nb_elems(x) + vector_nb_elems(yT),
                                3 * A->rows * A->cols);
    if (!stats)
        return NULL;

    vector_transpose(yT);

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas2_dger(A->rows, A->cols, alpha, A->data, x->data, yT->data);
                }
                else {
                    blas2_dger(A->rows, A->cols, alpha, A->data, x->data, yT->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dgemm(config_t cfg, double alpha, matrix_t* A, matrix_t* B, double beta,
                      matrix_t* C)
{
    stats_t* stats = stats_init("dgemm", 3, cfg.nb_threads,
                                matrix_nb_elems(A) + matrix_nb_elems(B) + matrix_nb_elems(C),
                                3 * A->rows * B->cols * (2 * B->rows));
    if (!stats)
        return NULL;

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas3_dgemm(A->rows, B->cols, B->rows, alpha, A->data, B->data, beta,
                                         C->data);
                }
                else {
                    blas3_dgemm(A->rows, B->cols, B->rows, alpha, A->data, B->data, beta, C->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    stats_compute(stats);
    return stats;
}

stats_t* driver_dgemm_var(config_t cfg, double alpha, matrix_t* A, matrix_t* B, double beta,
                          matrix_t* C)
{
    stats_t* stats = stats_init("dgemm_var", 3, cfg.nb_threads,
                                matrix_nb_elems(A) + matrix_nb_elems(B) + matrix_nb_elems(C),
                                2 * 3 * A->rows * B->cols * (2 * B->rows));
    if (!stats)
        return NULL;

    matrix_t* AT = matrix_copy(A);
    if (!AT)
        return NULL;
    matrix_transpose(A, AT);
    matrix_t* BT = matrix_copy(B);
    if (!BT)
        return NULL;
    matrix_transpose(B, BT);

    double elapsed;
    if (cfg.nb_threads != 1) {
        omp_set_num_threads(cfg.nb_threads);
    }
    for (size_t i = 0; i < MAX_SAMPLES; ++i) {
        do {
            instant_t start = instant_now();
            for (size_t _ = 0; _ < cfg.nb_reps; ++_) {
                if (cfg.nb_threads != 1) {
                    parallel_blas3_dgemm(A->rows, BT->cols, BT->rows, alpha, A->data, BT->data, 0,
                                         C->data);
                    parallel_blas3_dgemm(AT->rows, B->cols, B->rows, beta, AT->data, B->data, 0,
                                         C->data);
                }
                else {
                    blas3_dgemm(A->rows, BT->cols, BT->rows, alpha, A->data, BT->data, 0, C->data);
                    blas3_dgemm(AT->rows, B->cols, B->rows, beta, AT->data, B->data, 0, C->data);
                }
            }
            instant_t stop = instant_now();
            elapsed = compute_avg_latency(start, stop, cfg.nb_reps);
        } while (elapsed <= 0.0);
        stats->samples[i] = elapsed;
    }

    matrix_deinit(AT);
    matrix_deinit(BT);

    stats_compute(stats);
    return stats;
}
