MATRIX_OPERATIONS_DIRECTORY = ./../../MatrixOperations
CC = gcc

CFLAGS = -I $(MATRIX_OPERATIONS_DIRECTORY) -g -Wall
LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) -llapacke -lblas -lm
#LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) 'pkg-config --libs gsl'

OBJ = matrixOperations1.o matrixOperations2.o matrixOperations3.o matrixOperations4.o matrixOperations5.o

%.o : $(MATRIX_OPERATIONS_DIRECTORY)/%.c $(MATRIX_OPERATIONS_DIRECTORY)/matrixOperations.h Makefile
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all : matrixTest

matrixTest : matrixTest.c $(OBJ) Makefile
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) matrixTest.c -o matrixTest -lm

clean :
	rm -f *.o matrixTest
