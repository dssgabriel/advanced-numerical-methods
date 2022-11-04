#pragma once

#include "matrix.h"
#include "stats.h"

stats_t* driver_cgs(size_t n, matrix_t* A, vector_t* x, size_t deg_m, size_t reps);
stats_t* driver_mgs(size_t n, matrix_t* A, vector_t* x, size_t deg_m, size_t reps);