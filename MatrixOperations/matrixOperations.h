#ifndef __matrixOperations__
#define __matrixOperations__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>

#define precision double
#define UNDEFINED 0
#define ZEROS 1
#define ONES 2
#define FILL 3
#define IDENTITY 4
#define NOT_TRANSPOSED 0
#define TRANSPOSED 1
#define HORZ 0
#define VERT 1
#define COLOR 0
#define GRAYSCALE 1
#define PARENT 0
#define SUBMATRIX 1
#define IS_COLOR GRAYSCALE


typedef struct {
	precision *data;
	int numRows;
	int numCols;
	int span;
	int type;
} matrix_t;

#define elem(M, i, j) (M)->data[(j) * (M)->numRows + (i)]

#endif
