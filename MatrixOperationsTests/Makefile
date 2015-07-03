MATRIX_OPERATIONS_DIRECTORY = ../MatrixOperations
CC = gcc

CFLAGS = -I $(MATRIX_OPERATIONS_DIRECTORY) -g -Wall
LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY)
#LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) 'pkg-config --libs gsl'

OBJ = matrixOperations.o

%.o : $(MATRIX_OPERATIONS_DIRECTORY)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all : matrixTest0 


matrixTest0 : matrixTest0.c $(OBJ)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) matrixTest0.c -o matrixTest0

clean :
	rm -f *.o matrixTest0
