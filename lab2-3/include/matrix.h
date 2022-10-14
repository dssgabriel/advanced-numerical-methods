#pragma once

#include <stddef.h>

#define ALIGNMENT 64

/**
 * Represents a matrix storing double precision floating-point values stored
 * contiguously in memory, of dimensions `rows * cols`.
 **/
typedef struct matrix_s {
    double* data;
    size_t rows;
    size_t cols;
} matrix_t;

/**
 * Represents a vector storing double precision floating-point values.
 * Actually just a matrix with one of its dimensions set to 1.
 **/
typedef matrix_t vector_t;

matrix_t* matrix_read(const char* filename);

/**
 * Creates a new column vector of `len` elements, initialized with zeroes.
 **/
vector_t* vector_zeroes(size_t len);

/**
 * Creates a new matrix of `rows * cols` elements, initialized with zeroes.
 **/
matrix_t* matrix_zeroes(size_t rows, size_t cols);

/**
 * Creates a new column vector of `len` elements, initialized with ones.
 **/
vector_t* vector_ones(size_t len);

/**
 * Creates a new matrix of `rows * cols` elements, initialized with ones.
 **/
matrix_t* matrix_ones(size_t rows, size_t cols);

/**
 * Creates a new column vector of `len` elements, initialized with random values
 * ranging in the interval of ]-1.0, 1.0[.
 **/
vector_t* vector_rand_init(size_t len);

/**
 * Creates a new matrix of `rows * cols` elements, initialized with random
 * values ranging in the interval of ]-1.0, 1.0[.
 **/
matrix_t* matrix_rand_init(size_t rows, size_t cols);

/**
 * Deallocates a vector.
 **/
void vector_deinit(vector_t* self);

/**
 * Deallocates a matrix.
 **/
void matrix_deinit(matrix_t* self);

/**
 * Transposes a vector.
 **/
void vector_transpose(vector_t* self);

/**
 * Transposes a matrix.
 **/
void matrix_transpose(matrix_t const* self, matrix_t* transposed);

/**
 * Prints a given vector.
 **/
void print_vector(vector_t const* self, char const* name);

/**
 * Prints a given matrix.
 **/
void print_matrix(matrix_t const* self, char const* name);

size_t matrix_nb_elems(matrix_t const* self);

size_t vector_nb_elems(vector_t const* self);

matrix_t* matrix_copy(matrix_t const* self);
