#define precision float
#define UNDEFINED 0
#define ZEROS 1
#define NOT_TRANSPOSED 0
#define TRANSPOSED 1
#define HORZ 0
#define VERT 1
#define COLOR 0
#define GRAYSCALE 1
#define IS_COLOR GRAYSCALE

#include <stdint.h>

typedef struct {
	precision **data;
	uint64_t numRows;
	uint64_t numCols;
} matrix_t;

matrix_t * initializeMatrix (int mode, int sizeX, int sizeY);
void freeMatrix (matrix_t *M);
void fprintMatrix (FILE *stream, matrix_t *M);
void fwriteMatrix (FILE *stream, matrix_t *M);
matrix_t * fscanMatrix (FILE *stream);
matrix_t * freadMatrix (FILE *stream);
void subtractMatrixColumn (matrix_t *M1, int col10, matrix_t *M2, int col2);
matrix_t * matrixMultiply (matrix_t *A, int tranA, matrix_t *B, int tranB, uint64_t maxCol);
matrix_t * copyMatrix (matrix_t *M);
matrix_t * calcEigenvectorsSymmetric (matrix_t *M);
matrix_t * calcSurrogateMatrix (matrix_t *A);

void loadPPMtoMatrixCol (char *path, matrix_t *M, uint64_t col, unsigned char *pixels);
void writePPMgrayscale (char * filename, matrix_t *M, uint64_t COL, uint64_t rows, uint64_t cols);
matrix_t * calcMeanCol (matrix_t *A);
matrix_t * transposeMatrix (matrix_t *M);
