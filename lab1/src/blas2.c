#include "blas2.h"

#include <assert.h>

void blas2_dgemv(size_t m, size_t n, double alpha, double const* restrict A,
                 double const* restrict x, double beta, double* restrict y)
{
    if (!A || !x || !y)
        return;
    assert((m != 0 && n != 0) && "`m` and `n` must be different than 0.");

    for (size_t i = 0; i < m; ++i) {
        double tmp = 0.0;
        for (size_t j = 0; j < n; ++j) {
            tmp += A[i * n + j] * x[j];
        }
        y[i] = alpha * tmp + beta * y[i];
    }
}

void parallel_blas2_dgemv(size_t m, size_t n, double alpha, double const* restrict A,
                          double const* restrict x, double beta, double* restrict y)
{
    if (!A || !x || !y)
        return;
    assert((m != 0 && n != 0) && "`m` and `n` must be different than 0.");

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        double tmp = 0.0;
        for (size_t j = 0; j < n; ++j) {
            tmp += A[i * n + j] * x[j];
        }
        y[i] = alpha * tmp + beta * y[i];
    }
}

void blas2_dger(size_t m, size_t n, double alpha, double* restrict A, double const* restrict x,
                double const* restrict yT)
{
    if (!A || !x || !yT)
        return;
    assert((m != 0 && n != 0) && "`m` and `n` must be different than 0.");

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] += alpha * x[i] * yT[j];
        }
    }
}

void parallel_blas2_dger(size_t m, size_t n, double alpha, double* restrict A,
                         double const* restrict x, double const* restrict yT)
{
    if (!A || !x || !yT)
        return;
    assert((m != 0 && n != 0) && "`m` and `n` must be different than 0.");

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] += alpha * x[i] * yT[j];
        }
    }
}