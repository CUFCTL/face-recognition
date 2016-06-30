CC = gcc
CFLAGS = -g -Wall
LFLAGS = -lm -lblas -llapacke

INCS = src/database.h src/matrix.h src/ppm.h
OBJS = database.o matrix.o pca.o ppm.o
BINS = test-matrix test-ppm train recognize

all: $(BINS)

ppm.o: src/ppm.h src/ppm.c
	$(CC) -c $(CFLAGS) src/ppm.c -o $@

matrix.o: ppm.o src/matrix.h src/matrix.c
	$(CC) -c $(CFLAGS) src/matrix.c -o $@

database.o: ppm.o matrix.o src/database.h src/database.c
	$(CC) -c $(CFLAGS) src/database.c -o $@

pca.o: matrix.o src/database.h src/pca.c
	$(CC) -c $(CFLAGS) src/pca.c -o $@

lda.o: matrix.o src/database.h src/lda.c
	$(CC) -c $(CFLAGS) src/lda.c -o $@

ica.o: matrix.o src/database.h src/ica.c
	$(CC) -c $(CFLAGS) src/ica.c -o $@

test-matrix: matrix.o src/test_matrix.c
	$(CC) $(CFLAGS) matrix.o $(LFLAGS) src/test_matrix.c -o $@

test-ppm: ppm.o matrix.o src/test_ppm.c
	$(CC) $(CFLAGS) ppm.o matrix.o $(LFLAGS) src/test_ppm.c -o $@

train: ppm.o matrix.o database.o pca.o lda.o ica.o src/train.c
	$(CC) $(CFLAGS) ppm.o matrix.o database.o pca.o lda.o ica.o $(LFLAGS) src/train.c -o $@

recognize: ppm.o matrix.o database.o pca.o lda.o ica.o src/recognize.c
	$(CC) $(CFLAGS) ppm.o matrix.o database.o pca.o lda.o ica.o $(LFLAGS) src/recognize.c -o $@

clean:
	rm -f *.o *.dat $(BINS)
