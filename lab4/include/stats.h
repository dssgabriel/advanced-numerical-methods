#pragma once

#include <stddef.h>
#include <stdint.h>

#define MAX_SAMPLES 31

typedef struct stats_s {
    char* title;
    size_t size;
    double samples[MAX_SAMPLES];
    double min;
    double mean;
    double max;
    double median;
    double stddevp;
    double resQ;
    double resH;
} stats_t;

stats_t* stats_init(char const* title, size_t size);
void stats_deinit(stats_t* self);
void stats_compute(stats_t* self);
int stats_dump(stats_t const* self, char const* filename);
