#pragma once

#include "config.h"
#include "matrix.h"
#include "stats.h"

stats_t* driver_daxpy(config_t cfg, double alpha, vector_t* x, vector_t* y);
stats_t* driver_ddot(config_t cfg, vector_t* x, vector_t* y);
stats_t* driver_dnrm2(config_t cfg, vector_t* x);
stats_t* driver_dmax(config_t cfg, vector_t* x);

stats_t* driver_dgemv(config_t cfg, double alpha, matrix_t* A, vector_t* x, double beta, vector_t* y);
stats_t* driver_dgemv_var(config_t cfg, double alpha, matrix_t* A, vector_t* x, double beta, vector_t* y);
stats_t* driver_dger(config_t cfg, double alpha, matrix_t* A, vector_t* x, vector_t* yT);

stats_t* driver_dgemm(config_t cfg, double alpha, matrix_t* A, matrix_t* B, double beta, matrix_t* C);
stats_t* driver_dgemm_var(config_t cfg, double alpha, matrix_t* A, matrix_t* B, double beta, matrix_t* C);
