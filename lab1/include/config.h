#pragma once

#include "utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DEFAULT_LEN 65536
#define DEFAULT_THREADS 8
#define DEFAULT_REPS 1000

typedef enum blas_level_e {
    BLAS_ONE,
    BLAS_TWO,
    BLAS_THREE,
    BLAS_ONE_TWO,
    BLAS_ONE_THREE,
    BLAS_TWO_THREE,
    BLAS_ALL,
} blas_level_t;

typedef struct config_s {
    bool is_verbose;
    blas_level_t blas_level;
    size_t nb_threads;
    size_t nb_reps;
    char* output_filename;
    union {
        size_t len;
        struct {
            size_t cols;
            size_t rows;
        } pair;
    };
} config_t;

void help(char const* bin);
config_t config_init();
config_t config_from(int argc, char* argv[argc + 1]);
void config_print(config_t self);
