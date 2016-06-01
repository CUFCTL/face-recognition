CC = gcc
CFLAGS = -g -Wall
LFLAGS = -lm -lblas -llapacke

INCS = src/database.h src/matrix.h
OBJS = database.o matrix.o pca.o
BINS = test-matrix test-ppm train recognize

all: $(BINS)

matrix.o: src/matrix.h src/matrix.c
	$(CC) -c $(CFLAGS) src/matrix.c -o $@

database.o: matrix.o src/database.h src/database.c
	$(CC) -c $(CFLAGS) src/database.c -o $@

pca.o: matrix.o src/database.h src/pca.c
	$(CC) -c $(CFLAGS) src/pca.c -o $@

test-matrix: matrix.o src/test_matrix.c
	$(CC) $(CFLAGS) matrix.o $(LFLAGS) src/test_matrix.c -o $@

test-ppm: matrix.o src/test_ppm.c
	$(CC) $(CFLAGS) matrix.o $(LFLAGS) src/test_ppm.c -o $@

train: matrix.o database.o pca.o src/train.c
	$(CC) $(CFLAGS) matrix.o database.o pca.o $(LFLAGS) src/train.c -o $@

recognize: matrix.o database.o pca.o src/recognize.c
	$(CC) $(CFLAGS) matrix.o database.o pca.o $(LFLAGS) src/recognize.c -o $@

clean:
	rm -f *.o *.dat $(BINS)
