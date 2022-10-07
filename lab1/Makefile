CC := gcc
CFLAGS := -Wall -Wextra -Wno-format-overflow -g -fopenmp -I include/ -fno-omit-frame-pointer
OFLAGS := -march=native -Ofast -finline-functions -funroll-loops -ftree-vectorize -floop-interchange -fpeel-loops
LFLAGS := -lm

SRC := src
TARGET := target
DEPS := $(TARGET)/deps
BIN := $(TARGET)/mini-blas

.PHONY: all run build clean

run: build
	$(BIN)

build: $(DEPS)/config.o $(DEPS)/utils.o $(DEPS)/matrix.o $(DEPS)/drivers.o $(DEPS)/stats.o $(DEPS)/blas1.o $(DEPS)/blas2.o $(DEPS)/blas3.o $(DEPS)/main.o
	$(CC) $(CFLAGS) $(OFLAGS) $? -o $(BIN) $(LFLAGS)

$(DEPS)/%.o: $(SRC)/%.c
	@mkdir -p $(DEPS)
	$(CC) $(CFLAGS) $(OFLAGS) -c $? -o $@ $(LFLAGS)

clean:
	@rm -rf target
