#include "blas3.h"

#include <assert.h>

void blas3_dgemm(size_t l, size_t m, size_t n, double alpha, double const* restrict A,
                 double const* restrict B, double beta, double* restrict C)
{
    if (!A || !B || !C)
        return;
    assert((l != 0 && m != 0 && n != 0) && "`l`, `m` and `l` must be different than 0.");

    // if `alpha` and/or `beta` are zero
    if (alpha == 0.0) {
        if (beta == 0.0) {
            for (size_t i = 0; i < l; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    C[i * l + j] = 0.0;
                }
            }
        }
        else {
            for (size_t i = 0; i < l; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    C[i * l + j] *= beta;
                }
            }
        }
        return;
    }

    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double tmp = 0.0;
            for (size_t k = 0; k < n; ++k) {
                tmp += A[i * n + k] * B[k * m + j];
            }
            C[i * l + j] *= beta + alpha * tmp;
        }
    }
}

void parallel_blas3_dgemm(size_t l, size_t m, size_t n, double alpha, double const* restrict A,
                          double const* restrict B, double beta, double* restrict C)
{
    if (!A || !B || !C)
        return;
    assert((l != 0 && m != 0 && n != 0) && "`l`, `m` and `l` must be different than 0.");

    // if `alpha` and/or `beta` are zero
    if (alpha == 0.0) {
        if (beta == 0.0) {
#pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < l; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    C[i * l + j] = 0.0;
                }
            }
        }
        else {
#pragma omp parallel for collapse(2) schedule(static)
            for (size_t i = 0; i < l; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    C[i * l + j] *= beta;
                }
            }
        }
        return;
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double tmp = 0.0;
            for (size_t k = 0; k < n; ++k) {
                tmp += A[i * n + k] * B[k * m + j];
            }
            C[i * l + j] *= beta + alpha * tmp;
        }
    }
}