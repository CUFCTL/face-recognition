#include "matrixOperations.h"
// You need to install these to get it to work
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_eigen.h>

// Group 6 - other, doesn;t really fit in anywhere
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors);
matrix_t * m_getSubMatrix (matrix_t *M, int startRow, int startCol, int numRows, int numCols);

void loadPPMtoMatrixCol (char *path, matrix_t *M, int specCol, unsigned char *pixels);

void writePPMgrayscale (char *filename, matrix_t *M, int specCOl, int height, int width);


