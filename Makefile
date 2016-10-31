CC = gcc
CFLAGS = -g -Wall
LFLAGS = -lm -lblas -llapacke

INCS = src/database.h src/image.h src/image_entry.h src/matrix.h
OBJS = database.o image.o image_entry.o matrix.o pca.o lda.o ica.o
BINS = face-rec test-matrix test-image

all: $(BINS)

image.o: src/image.h src/image.c
	$(CC) -c $(CFLAGS) src/image.c -o $@

image_entry.o: src/image_entry.h src/image_entry.c
	$(CC) -c $(CFLAGS) src/image_entry.c -o $@

matrix.o: image.o src/matrix.h src/matrix.c
	$(CC) -c $(CFLAGS) src/matrix.c -o $@

database.o: image.o image_entry.o matrix.o src/database.h src/database.c
	$(CC) -c $(CFLAGS) src/database.c -o $@

pca.o: matrix.o src/database.h src/pca.c
	$(CC) -c $(CFLAGS) src/pca.c -o $@

lda.o: matrix.o src/database.h src/lda.c
	$(CC) -c $(CFLAGS) src/lda.c -o $@

ica.o: matrix.o src/database.h src/ica.c
	$(CC) -c $(CFLAGS) src/ica.c -o $@

#deleted ica.o out of each line
face-rec: image.o image_entry.o matrix.o database.o pca.o lda.o ica.o src/main.c
	$(CC) $(CFLAGS) image.o image_entry.o matrix.o database.o pca.o lda.o ica.o $(LFLAGS) src/main.c -o $@

test-image: image.o matrix.o src/test_image.c
	$(CC) $(CFLAGS) image.o matrix.o $(LFLAGS) src/test_image.c -o $@

test-matrix: matrix.o src/test_matrix.c
	$(CC) $(CFLAGS) matrix.o $(LFLAGS) src/test_matrix.c -o $@

clean:
	rm -f *.o *.dat *.csv $(BINS)
	rm -rf test_images train_images
