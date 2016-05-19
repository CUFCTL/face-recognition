MATRIX_PATH = ../MatrixOperations

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrixOperations.o
BINS = pca-train pca-recognize

%.o: $(MATRIX_PATH)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

pca-train: $(OBJS) pca_train.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pca_train.c -lm -llapacke -o pca-train

pca-recognize: $(OBJS) pca_recognize.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pca_recognize.c -lm -llapacke -o pca-recognize

clean:
	rm -f *.o *.dat *.ppm $(BINS)
