#pragma once

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
void blas1_daxpy(size_t len, double a, double const* x, double* y);

/**
 * Computes a double precision scalar-vector product and adds the result to a
 * vector in parallel using OpenMP.
 *
 * The `daxpy` routine performs a vector-vector operation defined as:
 *   y = a * x + y
 *
 * Where:
 * - `a` is a scalar.
 * - `x` and `y` are vectors.
 * - `len` is the number of elements in the vectors.
 **/
void parallel_blas1_daxpy(size_t len, double a, double const* x, double* y);

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
double blas1_ddot(size_t len, double const* x, double const* y);

/**
 * Computes a double precision vector-vector dot product in parallel using
 * OpenMP.
 *
 * The `ddot` routine performs a vector-vector operation defined as:
 *   res = x * y
 *
 * Where:
 * - `x` and `y` are vectors.
 * - `len` is the number of elements in the vectors.
 **/
double parallel_blas1_ddot(size_t len, double const* x, double const* y);

/**
 * Computes the double precision Euclidean/L2 norm of a vector.
 *
 * The `dnrm1` routine performs a vector reduction operation defined as:
 *   res = ||x||
 *
 * Where:
 * - `x` is a vector.
 * - `len` is the number of elements in the vector.
 **/
double blas1_dnrm2(size_t len, double const* x);

/**
 * Computes the double precision Euclidean/L2 norm of a vector in parallel using
 * OpenMP.
 *
 * The `dnrm1` routine performs a vector reduction operation defined as:
 *   res = ||x||
 *
 * Where:
 * - `x` is a vector.
 * - `len` is the number of elements in the vector.
 **/
double parallel_blas1_dnrm2(size_t len, double const* x);

/**
 * Finds the double precision maximum element of a vector.
 *
 * The `dnrm1` routine performs a vector reduction operation defined as:
 *   res = max(x)
 *
 * Where:
 * - `x` is a vector.
 * - `len` is the number of elements in the vector.
 *
 * The implementation of this routine is not branchless, but with compiler
 * optimizations enabled, the generated code is unrolled and uses
 * architecture-specific instructions that perform the comparisons without any
 * branching.
 **/
double blas1_dmax(size_t len, double const* x);

/**
 * Finds the double precision maximum element of a vector in parallel using
 * OpenMP.
 *
 * The `dnrm1` routine performs a vector reduction operation defined as:
 *   res = max(x)
 *
 * Where:
 * - `x` is a vector.
 * - `len` is the number of elements in the vector.
 *
 * The implementation of this routine is not branchless, but with compiler
 * optimizations enabled, the generated code is unrolled and uses
 * architecture-specific instructions that perform the comparisons without any
 * branching.
 **/
double parallel_blas1_dmax(size_t len, double const* x);
