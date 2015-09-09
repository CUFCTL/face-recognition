#include "matrixOperations.h"
#include "matrixOperations1.h"
#include "matrixOperations2.h"
#include "matrixOperations6.h"

// TEMPORARILY INCLUDED WHILE TESTING
void m_inverseMatrix (matrix_t *M);


// Group 3 - complex linear algebra functions of a single matrix
precision m_norm (matrix_t *M, int specRow);
matrix_t * m_sqrtm (matrix_t *M);
precision m_determinant (matrix_t *M);
matrix_t * m_cofactor (matrix_t *M);
matrix_t * m_covariance (matrix_t *M);


