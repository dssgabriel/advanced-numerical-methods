#include "config.h"
#include "drivers.h"
#include "matrix.h"
#include "stats.h"
#include "utils.h"

#include <assert.h>
#include <cblas.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int blas1_runs(config_t cfg)
{
    printf("\n──── BLAS 1 ────\n");
    printf("Input `x` and `y` vectors length: ");
    size_t len = read_user_input();
    len = len != 0 ? len : DEFAULT_LEN;

    double alpha = rand_double_range(-1.0, 1.0);
    vector_t* x = vector_rand_init(len);
    vector_t* y = vector_rand_init(len);
    if (!x || !y) {
        fprintf(stderr, BOLD RED "error:" RESET " failed vector allocation.\n");
        exit(EXIT_FAILURE);
    }

    if (cfg.is_verbose && len <= 12) {
        printf("Vectors and constant:\n");
        printf("alpha = %lf\n", alpha);
        print_vector(x, "x");
        print_vector(y, "y");
    }

    printf("Running...\n");
    stats_t* daxpy_stats = driver_daxpy(cfg, alpha, x, y);
    stats_t* ddot_stats = driver_ddot(cfg, x, y);
    stats_t* dnrm2_stats = driver_dnrm2(cfg, x);
    stats_t* dmax_stats = driver_dmax(cfg, x);
    stats_dump(daxpy_stats, cfg.output_filename);
    stats_dump(ddot_stats, cfg.output_filename);
    stats_dump(dnrm2_stats, cfg.output_filename);
    stats_dump(dmax_stats, cfg.output_filename);

    // Deallocate vectors
    vector_deinit(x);
    vector_deinit(y);

    return 0;
}

int blas2_runs(config_t cfg)
{
    printf("\n──── BLAS 2 ────\n");
    printf("Input `x` and `y` vectors length and `A` size: ");
    size_t len = read_user_input();
    len = len != 0 ? len : DEFAULT_LEN;

    double alpha = rand_double_range(-1.0, 1.0);
    double beta = rand_double_range(-1.0, 1.0);
    matrix_t* A = matrix_rand_init(len, len);
    vector_t* x = vector_rand_init(len);
    vector_t* y = vector_rand_init(len);
    if (!A || !x || !y) {
        return fprintf(stderr, BOLD RED "error:" RESET " failed matrix/vector allocation.\n") - 1;
    }

    if (cfg.is_verbose && len <= 12) {
        printf("Matrix, vectors & constants:\n");
        printf("alpha = %lf\n", alpha);
        printf("beta  = %lf\n", beta);
        print_matrix(A, "A");
        print_vector(x, "x");
        print_vector(y, "y");
    }

    printf("Running...\n");
    stats_t* dgemv_stats = driver_dgemv(cfg, alpha, A, x, beta, y);
    stats_t* dgemv_var_stats = driver_dgemv_var(cfg, alpha, A, x, beta, y);
    stats_t* dger_stats = driver_dger(cfg, alpha, A, x, y);
    stats_dump(dgemv_stats, cfg.output_filename);
    stats_dump(dgemv_var_stats, cfg.output_filename);
    stats_dump(dger_stats, cfg.output_filename);

    // Deallocate matrix and vectors
    matrix_deinit(A);
    vector_deinit(x);
    vector_deinit(y);

    return 0;
}

int blas3_runs(config_t cfg)
{
    printf("\n──── BLAS 3 ────\n");
    printf("Input `A`, `B` and `C` matrices size: ");
    size_t len = read_user_input();

    double alpha = rand_double_range(-1.0, 1.0);
    double beta = rand_double_range(-1.0, 1.0);
    matrix_t* A = matrix_rand_init(len, len);
    matrix_t* B = matrix_rand_init(len, len);
    matrix_t* C = matrix_ones(len, len);
    if (!A || !B || !C) {
        return fprintf(stderr, BOLD RED "error:" RESET " failed matrix allocation.\n") - 1;
    }

    if (cfg.is_verbose && len <= 12) {
        printf("Matrix, vectors & constants:\n");
        printf("alpha = %lf\n", alpha);
        printf("beta  = %lf\n", beta);
        print_matrix(A, "A");
        print_matrix(B, "B");
        print_matrix(C, "C");
    }

    printf("Running...\n");
    stats_t* dgemm_stats = driver_dgemm(cfg, alpha, A, B, beta, B);
    stats_t* dgemm_var_stats = driver_dgemm_var(cfg, alpha, A, B, beta, B);
    stats_dump(dgemm_stats, cfg.output_filename);
    stats_dump(dgemm_var_stats, cfg.output_filename);

    // Deallocate matrix and vectors
    matrix_deinit(A);
    matrix_deinit(B);
    matrix_deinit(C);

    return 0;
}

int main(int argc, char* argv[argc + 1])
{
    config_t cfg = (argc > 1) ? config_from(argc, argv) : config_init();
    if (cfg.is_verbose) {
        config_print(cfg);
    }

    FILE* ofp = cfg.output_filename != NULL ? fopen(cfg.output_filename, "wb") : stdout;
    if (!ofp)
        return -1;
    fprintf(ofp, "#%s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s\n", "title", "BLAS_lvl", "threads",
            "elems", "min", "mean", "max", "median", "stddevp", "GIB/s", "GFLOP/s");
    fclose(ofp);

    srand(0);
    if (cfg.blas_level == BLAS_ONE || cfg.blas_level == BLAS_ONE_TWO ||
        cfg.blas_level == BLAS_ONE_THREE || cfg.blas_level == BLAS_ALL) {
        blas1_runs(cfg);
    }

    if (cfg.blas_level == BLAS_TWO || cfg.blas_level == BLAS_ONE_TWO ||
        cfg.blas_level == BLAS_TWO_THREE || cfg.blas_level == BLAS_ALL) {
        blas2_runs(cfg);
    }

    if (cfg.blas_level == BLAS_THREE || cfg.blas_level == BLAS_ONE_THREE ||
        cfg.blas_level == BLAS_TWO_THREE || cfg.blas_level == BLAS_ALL) {
        blas3_runs(cfg);
    }

    return 0;
}
