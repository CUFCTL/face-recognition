MATRIX_OPERATIONS_DIRECTORY = ../MatrixOperations
CC = gcc

CFLAGS = -I $(MATRIX_OPERATIONS_DIRECTORY) -g -Wall
LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY)
#LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) 'pkg-config --libs gsl'

OBJ = matrixOperations.o

%.o : $(MATRIX_OPERATIONS_DIRECTORY)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all : pcaCreateDatabase pcaRecognition

pcaCreateDatabase : pcaCreateDatabase.c $(OBJ)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) pcaCreateDatabase.c -lm -llapacke -o pcaCreateDatabase

pcaRecognition : pcaRecognition.c $(OBJ)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) pcaRecognition.c -lm -llapacke -o pcaRecognition

clean :
	rm -f *.ppm *.o  *.dat pcaCreateDatabase pcaRecognition
