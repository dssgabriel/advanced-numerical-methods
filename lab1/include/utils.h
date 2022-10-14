#pragma once

#include <stddef.h>
#include <time.h>

#define ALIGNMENT 64
#define BUF_LEN 256
#define ONE_GIB 1073741824
#define MAX_PRINT_LEN 12

#define BOLD "\033[1m"
#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"

typedef struct timespec instant_t;

size_t read_user_input();
instant_t instant_now();
double compute_avg_latency(instant_t start, instant_t stop, size_t reps);
double rand_double_range(double min, double max);
void sort_double(double* data, size_t len);
double mean(double const* data, size_t len);
double stddev(double const* data, size_t len);
