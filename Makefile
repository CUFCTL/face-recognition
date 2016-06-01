CC = gcc
CFLAGS = -g -Wall

OBJS = matrix.o
BINS = test-matrix test-ppm train recognize

all: $(BINS)

matrix.o: src/matrix.h src/matrix.c
	$(CC) -c $(CFLAGS) src/matrix.c -o $@

test-matrix: $(OBJS) src/test_matrix.c
	$(CC) $(CFLAGS) $(OBJS) -lm -lblas -llapacke src/test_matrix.c -o $@

test-ppm: $(OBJS) src/test_ppm.c
	$(CC) $(CFLAGS) $(OBJS) -lm -lblas -llapacke src/test_ppm.c -o $@

train: $(OBJS) src/train.c
	$(CC) $(CFLAGS) $(OBJS) -lm -lblas -llapacke src/train.c -o $@

recognize: $(OBJS) src/recognize.c
	$(CC) $(CFLAGS) $(OBJS) -lm -lblas -llapacke src/recognize.c -o $@

clean:
	rm -f *.o *.dat $(BINS)
