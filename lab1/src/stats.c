#include "stats.h"

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

stats_t* stats_init(char const* title, uint8_t blas_lvl, size_t nb_threads, size_t nb_elems,
                    size_t nb_flops)
{
    stats_t* self = malloc(sizeof(stats_t));
    if (!self)
        return NULL;

    self->title = strdup(title);
    self->blas_lvl = blas_lvl;
    self->nb_threads = nb_threads;
    self->nb_elems = nb_elems;
    self->nb_bytes = nb_elems * (sizeof(double));
    self->nb_flops = nb_flops;

    return self;
}

void stats_compute(stats_t* self)
{
    if (!self)
        return;

    sort_double(self->samples, MAX_SAMPLES);
    self->min = self->samples[0];
    self->max = self->samples[MAX_SAMPLES - 1];
    self->mean = mean(self->samples, MAX_SAMPLES);
    if (MAX_SAMPLES & 1) {
        self->median = self->samples[(MAX_SAMPLES + 1) >> 1];
    }
    else {
        self->median =
            (self->samples[MAX_SAMPLES >> 1] + self->samples[(MAX_SAMPLES >> 1) + 1]) / 2.0;
    }
    self->stddevp = (stddev(self->samples, MAX_SAMPLES) * 100.0) / self->mean;

    double mean_s = self->mean / 1e9; // convert to seconds
    self->mem_throughput = (double)(self->nb_bytes) / (double)(ONE_GIB) / mean_s;
    self->ops_throughput = (self->nb_flops / 1e9) / mean_s;
}

int stats_dump(stats_t const* self, char const* filename)
{
    if (!self)
        return -1;

    FILE* ofp = (filename == NULL) ? stdout : fopen(filename, "ab");
    if (!ofp)
        return -1;

    fprintf(ofp, "%s; %u; %zu; %zu; %2.3lf; %2.3lf; %2.3lf; %2.3lf; %2.3lf%%; %2.3lf; %2.3lf\n",
            self->title, self->blas_lvl, self->nb_threads, self->nb_elems, self->min, self->mean,
            self->max, self->median, self->stddevp, self->mem_throughput, self->ops_throughput);

    if (filename) {
        fclose(ofp);
    }
    return 0;
}