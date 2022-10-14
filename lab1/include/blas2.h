#pragma once

#include <stddef.h>

/**
 * Computes a double precision matrix-vector product and adds the result to the
 * `y` vector.
 *
 * The `dgemv` routine performs a matrix-vector operation defined as:
 *   y = alpha * A * x + beta * y
 *
 * Where:
 * - `alpha` and `beta` are scalars.
 * - `x` and `y` are vectors.
 * - `A` is a matrix.
 * - `m` is the number of elements in the matrix rows.
 * - `n` is the number of elements in the matrix columns and in the vectors.
 **/
void blas2_dgemv(size_t m, size_t n, double alpha, double const* restrict A,
                 double const* restrict x, double beta, double* restrict y);

/**
 * Computes a double precision matrix-vector product and adds the result to the
 * `y` vector in parallel using OpenMP.
 *
 * The `dgemv` routine performs a matrix-vector operation defined as:
 *   y = alpha * A * x + beta * y
 *
 * Where:
 * - `alpha` and `beta` are scalars.
 * - `x` and `y` are vectors.
 * - `A` is a matrix.
 * - `m` is the number of elements in the matrix rows.
 * - `n` is the number of elements in the matrix columns and in the vectors.
 **/
void parallel_blas2_dgemv(size_t m, size_t n, double alpha, double const* restrict A,
                          double const* restrict x, double beta, double* restrict y);

/**
 * Performs a rank-1 update of a matrix.
 *
 * The `dger` routine performs a matrix-vector operation defined as:
 *   A = alpha * x * yT + A
 *
 * Where:
 * - `alpha` is a scalar.
 * - `x` and `yT` are vectors (`yT` is transposed).
 * - `A` is a matrix.
 * - `m` is the number of elements in the matrix rows.
 * - `n` is the number of elements in the matrix columns and in the vectors.
 **/
void blas2_dger(size_t m, size_t n, double alpha, double* restrict A, double const* restrict x,
                double const* restrict yT);

/**
 * Performs a rank-1 update of a matrix in parallel using OpenMP.
 *
 * The `dger` routine performs a matrix-vector operation defined as:
 *   A = alpha * x * yT + A
 *
 * Where:
 * - `alpha` is a scalar.
 * - `x` and `yT` are vectors (`yT` is transposed).
 * - `A` is a matrix.
 * - `m` is the number of elements in the matrix rows.
 * - `n` is the number of elements in the matrix columns and in the vectors.
 **/
void parallel_blas2_dger(size_t m, size_t n, double alpha, double* restrict A,
                         double const* restrict x, double const* restrict yT);
