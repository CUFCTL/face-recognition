MATRIX_PATH = ../../matrix

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrix.o
BINS = matrix-test

%.o: $(MATRIX_PATH)/%.c $(MATRIX_PATH)/%.h
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

matrix-test: $(OBJS) matrix_test.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -lm -lblas -llapacke matrix_test.c -o $@

clean:
	rm -f *.o *.log $(BINS)
