#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

inline size_t read_user_input()
{
    size_t input;
    scanf("%zu", &input);
    return input;
}

inline instant_t instant_now()
{
    instant_t self;
    clock_gettime(CLOCK_MONOTONIC_RAW, &self);
    return self;
}

inline double compute_avg_latency(instant_t start, instant_t stop, size_t reps)
{
    double s_in_ns = (double)(stop.tv_sec - start.tv_sec) * 1e9;
    double tot_ns = (stop.tv_nsec - start.tv_nsec) + s_in_ns;
    return tot_ns / (double)(reps);
}

inline double rand_double_range(double min, double max)
{
    double range = max - min;
    double div = (double)(RAND_MAX) / (double)(range);
    return min + random() / div;
}

void sort_double(double* data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        for (size_t j = i + 1; j < len; j++) {
            if (data[i] > data[j]) {
                double tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }
}

double mean(double const* data, size_t len)
{
    double tmp = 0.0;
    for (size_t i = 0; i < len; i++) {
        tmp += data[i];
    }
    return tmp / (double)(len);
}

double stddev(double const* data, size_t len)
{
    double dev = 0.0;
    double avg = mean(data, len);
    for (size_t i = 0; i < len; i++) {
        dev += (data[i] - avg) * (data[i] - avg);
    }
    dev /= (double)(len - 1);
    return sqrt(dev);
}
