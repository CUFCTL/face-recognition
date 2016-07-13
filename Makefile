CC = gcc
CFLAGS = -g -Wall
LFLAGS = -lm -lblas -llapacke

INCS = src/database.h src/image.h src/matrix.h
OBJS = database.o image.o matrix.o pca.o lda.o ica.o
BINS = face-rec test-matrix test-image

all: $(BINS)

image.o: src/image.h src/image.c
	$(CC) -c $(CFLAGS) src/image.c -o $@

matrix.o: image.o src/matrix.h src/matrix.c
	$(CC) -c $(CFLAGS) src/matrix.c -o $@

database.o: image.o matrix.o src/database.h src/database.c
	$(CC) -c $(CFLAGS) src/database.c -o $@

pca.o: matrix.o src/database.h src/pca.c
	$(CC) -c $(CFLAGS) src/pca.c -o $@

lda.o: matrix.o src/database.h src/lda.c
	$(CC) -c $(CFLAGS) src/lda.c -o $@

ica.o: matrix.o src/database.h src/ica.c
	$(CC) -c $(CFLAGS) src/ica.c -o $@

face-rec: image.o matrix.o database.o pca.o lda.o ica.o src/main.c
	$(CC) $(CFLAGS) image.o matrix.o database.o pca.o lda.o ica.o $(LFLAGS) src/main.c -o $@

test-image: image.o matrix.o src/test_image.c
	$(CC) $(CFLAGS) image.o matrix.o $(LFLAGS) src/test_image.c -o $@

test-matrix: matrix.o src/test_matrix.c
	$(CC) $(CFLAGS) matrix.o $(LFLAGS) src/test_matrix.c -o $@

clean:
	rm -f *.o *.dat $(BINS)
	rm -rf test_images train_images
