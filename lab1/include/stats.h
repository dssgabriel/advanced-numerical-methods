#pragma once

#include <stddef.h>
#include <stdint.h>

#define MAX_SAMPLES 31

typedef struct stats_s {
    char* title;
    uint8_t blas_lvl;
    size_t nb_threads;
    size_t nb_elems;
    size_t nb_bytes;
    size_t nb_flops;
    double samples[MAX_SAMPLES];
    double min;
    double mean;
    double max;
    double median;
    double stddevp;
    double mem_throughput;
    double ops_throughput;
} stats_t;

stats_t* stats_init(char const* title, uint8_t blas_lvl, size_t nb_threads, size_t nb_elems,
                    size_t nb_flops);

void stats_compute(stats_t* self);

int stats_dump(stats_t const* self, char const* filename);