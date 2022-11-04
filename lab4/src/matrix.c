#include "matrix.h"

#include "utils.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

matrix_t* matrix_read(char const* filename)
{
    if (!filename) {
        fprintf(stderr, "error: invalid argument, `filename` is null\n");
        exit(EXIT_FAILURE);
    }

    size_t rows = 0;
    size_t cols = 0;

    // Open file in read binary mode
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "error: failed to open file `%s`\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read matrix dimensions from file
    fscanf(fp, "%zu %zu", &rows, &cols);
    if (!rows || !cols) {
        fprintf(stderr, "error: failed to read from file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    matrix_t* self = matrix_zeroes(rows, cols);
    if (!self)
        return NULL;

    for (size_t i = 0; i < self->rows * self->cols; ++i) {
        // Read values from file
        fscanf(fp, "%lf ", &self->data[i]);
    }

    fclose(fp);
    return self;
}

vector_t* vector_zeroes(size_t len)
{
    return matrix_zeroes(len, 1);
}

matrix_t* matrix_zeroes(size_t rows, size_t cols)
{
    matrix_t* self = malloc(sizeof(matrix_t));
    if (!self)
        return NULL;

    self->rows = rows;
    self->cols = cols;
    self->data = aligned_alloc(ALIGNMENT, rows * cols * sizeof(double));
    if (!self->data) {
        free(self);
        return NULL;
    }

    memset(self->data, 0, rows * cols * sizeof(double));
    return self ? self : NULL;
}

vector_t* vector_ones(size_t len)
{
    return matrix_ones(len, 1);
}

matrix_t* matrix_ones(size_t rows, size_t cols)
{
    matrix_t* self = malloc(sizeof(matrix_t));
    if (!self)
        return NULL;

    self->rows = rows;
    self->cols = cols;
    self->data = aligned_alloc(ALIGNMENT, rows * cols * sizeof(double));
    if (!self->data) {
        free(self);
        return NULL;
    }

    for (size_t i = 0; i < rows * cols; ++i) {
        self->data[i] = 1.0;
    }

    return self;
}

vector_t* vector_rand_init(size_t len)
{
    vector_t* self = matrix_rand_init(len, 1);
    return self != NULL ? self : NULL;
}

matrix_t* matrix_rand_init(size_t rows, size_t cols)
{
    matrix_t* self = malloc(sizeof(matrix_t));
    if (!self)
        return NULL;

    self->rows = rows;
    self->cols = cols;
    self->data = aligned_alloc(ALIGNMENT, rows * cols * sizeof(double));
    if (!self->data) {
        free(self);
        return NULL;
    }

    for (size_t i = 0; i < rows * cols; ++i) {
        self->data[i] = rand_double_range(-1.0, 1.0);
    }

    return self;
}

void vector_deinit(vector_t* self)
{
    if (!self)
        return;
    matrix_deinit(self);
}

void matrix_deinit(matrix_t* self)
{
    if (self) {
        if (self->data) {
            free(self->data);
        }
        free(self);
    }
}

void vector_transpose(vector_t* self)
{
    if (!self)
        return;

    size_t tmp = self->rows;
    self->rows = self->cols;
    self->cols = tmp;
}

void matrix_transpose(matrix_t const* self, matrix_t* transposed)
{
    assert(self->rows == transposed->cols && self->cols == transposed->rows);

    double(*const restrict self_data)[self->cols] = (double(*)[self->cols])self->data;
    double(*restrict transposed_data)[transposed->cols] =
        (double(*)[transposed->cols])transposed->data;

    for (size_t i = 0; i < self->rows; ++i) {
        for (size_t j = 0; j < self->cols; ++j) {
            transposed_data[j][i] = self_data[i][j];
        }
    }
}

void print_vector(vector_t const* self, char const* name)
{
    if (!self)
        return;
    print_matrix(self, name);
}

void print_matrix(matrix_t const* self, char const* name)
{
    if (!self)
        return;

    char shift[32];
    size_t k;
    for (k = 0; k < (name != NULL ? strlen(name) + 3 : 4); ++k) {
        shift[k] = ' ';
    }
    shift[k] = '\0';

    printf("%s = ", name != NULL ? name : "M");
    for (size_t i = 0; i < self->rows; ++i) {
        if (self->rows == 1) {
            printf("⟮ ");
        }
        else if (i == 0) {
            printf("⎛ ");
        }
        else if (i != self->rows - 1) {
            printf("%s⎜ ", shift);
        }
        else {
            printf("%s⎝ ", shift);
        }

        for (size_t j = 0; j < self->cols; ++j) {
            if (self->data[i * self->cols + j] >= 0) {
                printf("%2.3lf ", self->data[i * self->cols + j]);
            }
            else {
                printf("%2.3lf ", self->data[i * self->cols + j]);
            }

            if (self->rows != 1) {
                if (i == 0 && j == self->cols - 1) {
                    printf("⎞\n");
                }
                else if (i == self->rows - 1 && j == self->cols - 1) {
                    printf("⎠\n");
                }
            }
        }

        if (self->rows == 1) {
            printf("⟯\n");
        }
        else if (i != 0 && i != self->rows - 1) {
            printf("⎟\n");
        }
    }
}

size_t matrix_nb_elems(matrix_t const* self)
{
    return self->rows * self->cols;
}

size_t vector_nb_elems(vector_t const* self)
{
    return matrix_nb_elems(self);
}

matrix_t* matrix_copy(matrix_t const* self)
{
    matrix_t* copy = matrix_zeroes(self->rows, self->cols);
    if (!copy) return NULL;

    for (size_t i = 0; i < self->rows * self->cols; ++i) {
        copy->data[i] = self->data[i];
    }

    return copy;
}
