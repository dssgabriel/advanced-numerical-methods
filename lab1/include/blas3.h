#pragma once

#include <stddef.h>

/**
 * Computes a double precision matrix-matrix product and adds the result to the
 * `C` matrix.
 *
 * The `dgemm` routine performs a matrix-matrix operation defined as:
 *   C = alpha * A * B + beta * C
 *
 * Where:
 * - `alpha` and `beta` are scalars.
 * - `A`, `B` and `C` are matrices.
 * - `l` is the number of elements in the rows of the `A` and `C` matrices.
 * - `m` is the number of elements in the columns of the `B` and `C` matrices.
 * - `n` is the number of elements in the columns of the `A` matrix and in the
 *   rows of the `B` matrix.
 **/
void blas3_dgemm(size_t l, size_t m, size_t n, double alpha, double const* restrict A,
                 double const* restrict B, double beta, double* restrict C);

/**
 * Computes a double precision matrix-matrix product and adds the result to the
 * `C` matrix in parallel using OpenMP;
 *
 * The `dgemm` routine performs a matrix-matrix operation defined as:
 *   C = alpha * A * B + beta * C
 *
 * Where:
 * - `alpha` and `beta` are scalars.
 * - `A`, `B` and `C` are matrices.
 * - `l` is the number of elements in the rows of the `A` and `C` matrices.
 * - `m` is the number of elements in the columns of the `B` and `C` matrices.
 * - `n` is the number of elements in the columns of the `A` matrix and in the
 *   rows of the `B` matrix.
 **/
void parallel_blas3_dgemm(size_t l, size_t m, size_t n, double alpha, double const* restrict A,
                          double const* restrict B, double beta, double* restrict C);