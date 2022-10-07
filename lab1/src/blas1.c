#include "blas1.h"

#include <math.h>

void blas1_daxpy(size_t len, double a, double const* x, double* y)
{
    for (size_t i = 0; i < len; ++i) {
        y[i] += a * x[i];
    }
}

void parallel_blas1_daxpy(size_t len, double a, double const* x, double* y)
{
    #pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
        y[i] += a * x[i];
    }
}

double blas1_ddot(size_t len, double const* x, double const* y)
{
    double res = 0.0;

    for (size_t i = 0; i < len; ++i) {
        res += x[i] * y[i];
    }

    return res;
}

double parallel_blas1_ddot(size_t len, double const* x, double const* y)
{
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for (size_t i = 0; i < len; ++i) {
        res += x[i] * y[i];
    }

    return res;
}

double blas1_dnrm2(size_t len, double const* x)
{
    double res = 0.0;

    for (size_t i = 0; i < len; ++i) {
        res += x[i] * x[i];
    }

    return sqrt(res);
}

double parallel_blas1_dnrm2(size_t len, double const* x)
{
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for (size_t i = 0; i < len; ++i) {
        res += x[i] * x[i];
    }

    return sqrt(res);
}

double blas1_dmax(size_t len, double const* x)
{
    double res = x[0];

    for (size_t i = 0; i < len; ++i) {
        res = (x[i] > res) ? x[i] : res;
    }

    return res;
}

double parallel_blas1_dmax(size_t len, double const* x)
{
    double res = x[0];

    #pragma omp parallel for reduction(max:res)
    for (size_t i = 0; i < len; ++i) {
        res = (x[i] > res) ? x[i] : res;
    }

    return res;
}
