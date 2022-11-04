#pragma once

#include "matrix.h"

#include <stddef.h>

/**
 * Computes a double precision scalar-vector product and adds the result to a
 * vector.
 *
 * The `daxpy` routine performs a vector-vector operation defined as:
 *   y = a * x + y
 *
 * Where:
 * - `a` is a scalar.
 * - `x` and `y` are vectors.
 * - `len` is the number of elements in the vectors.
 **/
void daxpy(size_t len, double alpha, double const* x, double* y);

/**
 * Computes a double precision vector-vector dot product.
 *
 * The `ddot` routine performs a vector-vector operation defined as:
 *   res = x * y
 *
 * Where:
 * - `x` and `y` are vectors.
 * - `len` is the number of elements in the vectors.
 **/
double ddot(size_t len, double const* x, double const* y);

/**
 * Computes the double precision Euclidean/L2 norm of a vector.
 * 
 * The `nrm2` routine performs a vector reduction operation defined as:
 *   res = ||x||
 *
 * Where:
 * - `x` is a vector.
 * - `len` is the number of elements in the vector.
 **/
double dnrm2(size_t len, double const* x);

/**
 * Computes the double precision Frobenius norm of a matrix.
 * 
 * The `nrmf` routine performs a matrix operation defined as:
 *   res = ||A||
 *
 * Where:
 * - `A` is a matrix.
 * - `rows` is the number of elements in the matrix's rows.
 * - `cols` is the number of elements in the matrix's cols.
 **/
double dnrmf(size_t rows, size_t cols, double const* A);

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
void dgemv(size_t m, size_t n, double alpha,
           double const* restrict A, double const* restrict x,
           double beta, double* restrict y);

void classical_gram_schmidt(size_t n, double* restrict x, double* restrict A,
                            size_t deg_m, matrix_t* mat_Q, matrix_t* mat_H);

void modified_gram_schmidt(size_t n, double* restrict x, double* restrict A,
                           size_t deg_m, matrix_t* mat_Q, matrix_t* mat_H);
