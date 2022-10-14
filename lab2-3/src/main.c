#include "drivers.h"
#include "matrix.h"
#include "stats.h"
#include "utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[argc + 1])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <SIZE> <DEGREE> <REPETITIONS> [OUTFILE]\n", argv[0]);
        return -1;
    }

    size_t const size = (size_t)(atoi(argv[1]));
    size_t const degree = (size_t)(atoi(argv[2]));
    size_t const reps = (size_t)(atoi(argv[3]));
    char* outfile = argc == 5 ? argv[4] : NULL;

    matrix_t* A = matrix_rand_init(size, size);
    vector_t* x = vector_zeroes(size);
    x->data[0] = 1.0;

    if (size == 4) {
        FILE* ofp = outfile != NULL ? fopen(outfile, "wb") : stdout;
        if (!ofp) {
            fprintf(stderr, "error: failed to open output file `%s`.", outfile);
            return -1;
        }
        fprintf(ofp, "#%s; %s; %s; %s; %s; %s; %s\n", "title", "size", "min", "mean", "max",
                "median", "stddevp");
        if (ofp != stdout)
            fclose(ofp);
    }

    stats_t* cgs = driver_cgs(size, A, x, degree, reps);
    stats_t* mgs = driver_mgs(size, A, x, degree, reps);

    double err_Q = compute_error(cgs->resQ, mgs->resQ);
    assert(err_Q <= ERR_TOL);
    double err_H = compute_error(cgs->resH, mgs->resH);
    assert(err_H <= ERR_TOL);

    stats_dump(cgs, outfile);
    stats_dump(mgs, outfile);

    stats_deinit(cgs);
    stats_deinit(mgs);
    matrix_deinit(A);
    vector_deinit(x);

    return 0;
}