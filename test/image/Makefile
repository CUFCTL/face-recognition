MATRIX_PATH = ../../matrix

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrix.o
BINS = ppm-test

%.o: $(MATRIX_PATH)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

ppm-test: $(OBJS) ppm_test.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -lm -llapacke ppm_test.c -o $@

clean:
	rm -f *.o *.dat *.ppm $(BINS)
