CC := gcc
CFLAGS := -Wall -Wextra -Wno-format-overflow -g -fopenmp -I include/ -fno-omit-frame-pointer
OFLAGS := -march=native -Ofast -finline-functions -funroll-loops -ftree-vectorize -fpeel-loops
LFLAGS := -lm

SRC := src
TARGET := target
DEPS := $(TARGET)/deps
BIN := $(TARGET)/arnoldi

.PHONY: run build clean

run: build
	$(BIN) 64 63 100

build: $(DEPS)/utils.o $(DEPS)/stats.o $(DEPS)/drivers.o $(DEPS)/matrix.o $(DEPS)/blas.o $(DEPS)/main.o
	$(CC) $(CFLAGS) $(OFLAGS) $? -o $(BIN) $(LFLAGS)

$(DEPS)/%.o: $(SRC)/%.c
	@mkdir -p $(DEPS)
	$(CC) $(CFLAGS) $(OFLAGS) -c $? -o $@ $(LFLAGS)

clean:
	@rm -rf target
