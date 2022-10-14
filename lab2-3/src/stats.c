#include "stats.h"

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

stats_t* stats_init(char const* title, size_t size)
{
    stats_t* self = malloc(sizeof(stats_t));
    if (!self)
        return NULL;

    self->title = strdup(title);
    self->size = size;
    return self;
}

void stats_deinit(stats_t* self)
{
    if (self) {
        if (self->title) {
            free(self->title);
        }
        free(self);
    }
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
}

int stats_dump(stats_t const* self, char const* filename)
{
    if (!self)
        return -1;

    FILE* ofp = (filename == NULL) ? stdout : fopen(filename, "ab");
    if (!ofp)
        return -1;

    fprintf(ofp, "%s; %zu; %2.3lf; %2.3lf; %2.3lf; %2.3lf; %2.3lf%%\n", self->title, self->size,
            self->min, self->mean, self->max, self->median, self->stddevp);

    if (filename) {
        fclose(ofp);
    }
    return 0;
}