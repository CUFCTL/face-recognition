MATRIX_PATH = ../MatrixOperations

CC = gcc
CFLAGS = -I $(MATRIX_PATH) -g -Wall
LFLAGS = -L $(MATRIX_PATH)

OBJS = matrixOperations.o
BINS = pcaCreateDatabase pcaRecognition

%.o: $(MATRIX_PATH)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all: $(BINS)

pcaCreateDatabase: $(OBJS) pcaCreateDatabase.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pcaCreateDatabase.c -lm -llapacke -o pcaCreateDatabase

pcaRecognition: $(OBJS) pcaRecognition.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) pcaRecognition.c -lm -llapacke -o pcaRecognition

clean:
	rm -f *.o *.dat *.ppm $(BINS)
