MATRIX_PATH = ../MatrixOperations

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrixOperations.o
BINS = pca-train pcaRecognition

%.o: $(MATRIX_PATH)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

pca-train: $(OBJS) pca_train.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pca_train.c -lm -llapacke -o pca-train

pcaRecognition: $(OBJS) pcaRecognition.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pcaRecognition.c -lm -llapacke -o pcaRecognition

clean:
	rm -f *.o *.dat *.ppm $(BINS)
