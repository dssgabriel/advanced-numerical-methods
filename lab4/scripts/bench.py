#!/usr/bin/python

import os

def main():
    matrix_size = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    reps = [20000, 10000, 5000, 500, 200, 50, 20, 10, 2]

    for (size, rep) in zip(matrix_size, reps):
        os.system(f"target/gram-schmidt {size} {size - 1} {rep}")

if __name__ == '__main__':
    main()
