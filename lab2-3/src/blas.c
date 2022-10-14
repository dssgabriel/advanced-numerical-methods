#include "blas.h"

#include "matrix.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

void daxpy(size_t len, double alpha, double const* x, double* y)
{
    for (size_t i = 0; i < len; ++i) {
        y[i] += alpha * x[i];
    }
}

double ddot(size_t len, double const* restrict x, double const* restrict y)
{
    double res = 0.0;

    for (size_t i = 0; i < len; ++i) {
        res += x[i] * y[i];
    }

    return res;
}

double dnrm2(size_t len, double const* restrict x)
{
    double res = 0.0;

    for (size_t i = 0; i < len; ++i) {
        res += x[i] * x[i];
    }

    return sqrt(res);
}

double dnrmf(size_t rows, size_t cols, double const* restrict A)
{
    double res = 0.0;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res += fabs(A[i * cols + j] * A[i * cols + j]);
        }
    }

    return sqrt(res);
}

void dgemv(size_t m, size_t n, double alpha, double const* restrict A, double const* restrict x,
           double beta, double* restrict y)
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

void classical_gram_schmidt(size_t n, double* restrict x, double* restrict A, size_t deg_m,
                            matrix_t* mat_Q, matrix_t* mat_H)
{
    double epsilon = 1e-12;
    double(*restrict H)[deg_m] = (double(*)[mat_H->cols])mat_H->data;
    double(*restrict Q)[deg_m] = (double(*)[mat_Q->cols])mat_Q->data;
    double* q_k = aligned_alloc(64, mat_Q->rows * sizeof(double));
    if (!q_k)
        return;
    double* v = aligned_alloc(64, n * sizeof(double));
    if (!v)
        return;
    double* q_j_T = aligned_alloc(64, mat_Q->rows * sizeof(double));
    if (!q_j_T)
        return;

// Normalize first vector
#pragma omp simd
    for (size_t _ = 0; _ < n; ++_) {
        Q[_][0] = x[_] / dnrm2(n, x);
    }

    for (size_t k = 1; k < deg_m; ++k) {
// Initialize Q[:,k-1] slice
#pragma omp simd
        for (size_t _ = 0; _ < mat_Q->rows; ++_) {
            q_k[_] = Q[_][k - 1];
        }
        // v_k+1 = A * v_k, where v_k = q_k and v = v_k+1
        dgemv(n, n, 1.0, A, q_k, 0.0, v);

        for (size_t j = 0; j < k; ++j) {
// Initialize Q[:,j] slice
#pragma omp simd
            for (size_t _ = 0; _ < mat_Q->rows; ++_) {
                q_j_T[_] = Q[_][j];
            }
            // h_j_k-1 = v_k+1 * v_k
            H[j][k - 1] = ddot(mat_Q->rows, q_j_T, v);
        }

        for (size_t j = 0; j < k; ++j) {
// Initialize Q[:,j] slice
#pragma omp simd
            for (size_t _ = 0; _ < mat_Q->rows; ++_) {
                q_j_T[_] = Q[_][j];
            }
            // v_k+1 -= h_j_k-1 * v_k
            daxpy(n, -H[j][k - 1], q_j_T, v);
        }

        H[k][k - 1] = dnrm2(n, v);
        if (H[k][k - 1] > epsilon) {
#pragma omp simd
            for (size_t _ = 0; _ < n; ++_) {
                Q[_][k] = v[_] / H[k][k - 1];
            }
        }
        else {
            goto cleanup;
        }
    }

cleanup:
    free(q_k);
    free(v);
    free(q_j_T);
}

void modified_gram_schmidt(size_t n, double* restrict x, double* restrict A, size_t deg_m,
                           matrix_t* mat_Q, matrix_t* mat_H)
{
    double epsilon = 1e-12;
    double(*restrict H)[deg_m] = (double(*)[mat_H->cols])mat_H->data;
    double(*restrict Q)[deg_m] = (double(*)[mat_Q->cols])mat_Q->data;
    double* q_k = aligned_alloc(64, mat_Q->rows * sizeof(double));
    if (!q_k)
        return;
    double* v = aligned_alloc(64, n * sizeof(double));
    if (!v)
        return;
    double* q_j_T = aligned_alloc(64, mat_Q->rows * sizeof(double));
    if (!q_j_T)
        return;

#pragma omp simd
    for (size_t _ = 0; _ < n; ++_) {
        Q[_][0] = x[_] / dnrm2(n, x);
    }

    for (size_t k = 1; k < deg_m; ++k) {
// Initialize Q[:,k-1] slice
#pragma omp simd
        for (size_t _ = 0; _ < mat_Q->rows; ++_) {
            q_k[_] = Q[_][k - 1];
        }
        // Candidate vector
        dgemv(n, n, 1.0, A, q_k, 0.0, v);

        for (size_t j = 0; j < k; ++j) {
// Initialize Q[:,j] slice (transposed but it's a vector so who fucking cares)
#pragma omp simd
            for (size_t _ = 0; _ < mat_Q->rows; ++_) {
                q_j_T[_] = Q[_][j];
            }

            H[j][k - 1] = ddot(mat_Q->rows, q_j_T, v);
            daxpy(n, -H[j][k - 1], q_j_T, v);
        }

        H[k][k - 1] = dnrm2(n, v);
        if (H[k][k - 1] > epsilon) {
#pragma omp simd
            for (size_t _ = 0; _ < n; ++_) {
                Q[_][k] = v[_] / H[k][k - 1];
            }
        }
        else {
            goto cleanup;
        }
    }

cleanup:
    free(q_k);
    free(v);
    free(q_j_T);
}
