MATRIX_OPERATIONS_DIRECTORY = ../MatrixOperations
CC = gcc

CFLAGS = -I $(MATRIX_OPERATIONS_DIRECTORY) -g -Wall
LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) -lm
#LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) 'pkg-config --libs gsl'

OBJ = matrixOperations.o

%.o : $(MATRIX_OPERATIONS_DIRECTORY)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all : matrixTest

matrixTest : matrixTest.c $(OBJ)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) matrixTest.c -o matrixTest -lm

clean :
	rm -f *.o matrixTest
