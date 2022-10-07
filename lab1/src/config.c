#include "config.h"
#include "utils.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void help(char const* bin)
{
    fprintf(stderr, "Parallel Mini-BLAS - MPNA\n\n");
    fprintf(stderr, BOLD "Usage: " RESET);
    fprintf(stderr, "%s [FLAGS] [OPTIONS]\n\n", bin);
    fprintf(stderr, BOLD "Flags:\n" RESET);
    fprintf(stderr, "  -h, --help                  Print help information.\n");
    fprintf(stderr, "  -v, --verbose               Be verbose.\n\n");
    fprintf(stderr, BOLD "Options:\n" RESET);
    fprintf(stderr, "  -1, --blas1                 Run BLAS1 routines.\n");
    fprintf(stderr, "  -2, --blas2                 Run BLAS2 routines.\n");
    fprintf(stderr, "  -3, --blas3                 Run BLAS3 routines.\n");
    fprintf(stderr, "  -a, --blas-all              Run all BLAS routines (default).\n");
    fprintf(stderr, "  -p, --parallel [NB_CORES]   Enable parallel mode and optionally specify number of threads.\n");
    fprintf(stderr, "  -r, --repetitions <NB_REPS> Specify number of repetitions of the BLAS kernel.\n");
    fprintf(stderr, "  -o, --output <FILENAME>     Specify the output filename (stdout by default).\n\n");
}

char *blas_level_to_str(blas_level_t blas_level)
{
    switch (blas_level) {
        case BLAS_ONE: return "BLAS1";
        case BLAS_TWO: return "BLAS2";
        case BLAS_THREE: return "BLAS3";
        case BLAS_ONE_TWO: return "BLAS1 & BLAS2";
        case BLAS_ONE_THREE: return "BLAS1 & BLAS3";
        case BLAS_TWO_THREE: return "BLAS2 & BLAS3";
        case BLAS_ALL: return "all BLAS levels";
    }

    // Unreachable
    return NULL;
}

config_t config_init()
{
    config_t self = {
        .is_verbose = false,
        .blas_level = BLAS_ALL,
        .nb_threads = 1,
        .nb_reps = DEFAULT_REPS,
        .pair = { DEFAULT_LEN, DEFAULT_LEN },
        .output_filename = NULL,
    };

    return self;
}

config_t config_from(int argc, char* argv[argc + 1])
{
    config_t self = config_init();
    int curr_opt = 0;

    for (;;) {
        static struct option long_opts[] = {
            { "help", no_argument, NULL, 'h' },
            { "verbose", no_argument, NULL, 'v' },
            { "blas1", no_argument, NULL, '1' },
            { "blas2", no_argument, NULL, '2' },
            { "blas3", no_argument, NULL, '3' },
            { "blas-all", no_argument, NULL, 'a' },
            { "parallel", optional_argument, NULL, 'p' },
            { "repetitions", required_argument, NULL, 'r' },
            { "output", required_argument, NULL, 'o' },
            { NULL, 0, NULL, 0 },
        };

        int opt_idx = 0;
        curr_opt = getopt_long(argc, argv, "hv123ap::r:o:", long_opts, &opt_idx);
        if (curr_opt == -1) break;

        switch (curr_opt) {
            case 'h':
                help(argv[0]);
                exit(EXIT_SUCCESS);

            case 'v':
                self.is_verbose = true;
                break;

            case 'p':
                if (optarg != NULL) {
                    self.nb_threads = (size_t)(atoi(optarg));
                } else {
                    self.nb_threads = DEFAULT_THREADS;
                }
                break;
            
            case 'r':
                self.nb_reps = (size_t)(atoi(optarg));
                break;

            case 'o':
                self.output_filename = strdup(optarg);
                break;

            case '1':
                if (self.blas_level == BLAS_TWO) {
                    self.blas_level = BLAS_ONE_TWO;
                } else if (self.blas_level == BLAS_THREE) {
                    self.blas_level = BLAS_ONE_THREE;
                } else if (self.blas_level == BLAS_TWO_THREE) {
                    self.blas_level = BLAS_ALL;
                } else {
                    self.blas_level = BLAS_ONE;
                }
                break;

            case '2':
                if (self.blas_level == BLAS_ONE) {
                    self.blas_level = BLAS_ONE_TWO;
                } else if (self.blas_level == BLAS_THREE) {
                    self.blas_level = BLAS_TWO_THREE;
                } else if (self.blas_level == BLAS_ONE_THREE) {
                    self.blas_level = BLAS_ALL;
                } else {
                    self.blas_level = BLAS_TWO;
                }
                break;

            case '3':
                if (self.blas_level == BLAS_ONE) {
                    self.blas_level = BLAS_ONE_THREE;
                } else if (self.blas_level == BLAS_TWO) {
                    self.blas_level = BLAS_TWO_THREE;
                } else if (self.blas_level == BLAS_ONE_TWO) {
                    self.blas_level = BLAS_ALL;
                } else {
                    self.blas_level = BLAS_THREE;
                }
                break;

            case 'a':
                self.blas_level = BLAS_THREE;
                break;

            default:
                fprintf(stderr,
                        BOLD RED "error:" RESET " unrecognized option.\n"
                        "See help below for available options.\n\n");
                help(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    return self;
}

void config_print(config_t self)
{
    printf(BOLD "Configuration:\n" RESET);
    printf("  verbose:           " BLUE "%s" RESET "\n", self.is_verbose == true ? "yes" : "no");
    printf("  BLAS level:        " BLUE "%s" RESET "\n", blas_level_to_str(self.blas_level));
    printf("  number of threads: " BLUE "%zu" RESET "\n", self.nb_threads);
    printf("  number of reps:    " BLUE "%zu" RESET "\n", self.nb_reps);
    printf("  output filename:   " BLUE "%s" RESET "\n", self.output_filename ? self.output_filename : "stdout");
}