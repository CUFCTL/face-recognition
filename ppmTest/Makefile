MATRIX_OPERATIONS_DIRECTORY = ../MatrixOperations
CC = gcc

CFLAGS = -I $(MATRIX_OPERATIONS_DIRECTORY) -g -Wall
LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY)
#LFLAGS = -L $(MATRIX_OPERATIONS_DIRECTORY) 'pkg-config --libs gsl'

OBJ = matrixOperations.o

%.o : $(MATRIX_OPERATIONS_DIRECTORY)/%.c
	$(CC) -c $(CFLAGS) $(LFLAGS) $< -o $@

all : ppmTest


ppmTest : ppmTest.c $(OBJ)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJ) ppmTest.c -o ppmTest

clean :
	rm -f *.ppm *.o  *.dat ppmTest
