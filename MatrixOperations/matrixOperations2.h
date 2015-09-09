#include "matrixOperations.h"
#include "matrixOperations1.h"

/***************** Group 2 - Operations on a single matrix *******************/
/***** 2.0 - No return values, operate directly on M's data *****/
// 2.0.0
//	- Not element wise operation
//	- no extra inputs
void m_flipCols (matrix_t *M);
void m_normalize (matrix_t *M);
// THIS IS INCLUDED FOR GROUP 3 FOR TESTING TEMPORARILY
//void m_inverseMatrix (matrix_t *M); // Must be square matrix

// 2.0.1
//	- element wise math operation
//	- no extra inputs
void m_elem_truncate (matrix_t *M);
void m_elem_acos (matrix_t *M);
void m_elem_sqrt (matrix_t *M);
void m_elem_negate (matrix_t *M);
void m_elem_exp (matrix_t *M);
// 2.0.2
//	- element wise math operation
//	- has a second input operation relies on
void m_elem_pow (matrix_t *M, precision x);
void m_elem_mult (matrix_t *M, precision x);
void m_elem_divideByConst (matrix_t *M, precision x);
void m_elem_divideByMatrix (matrix_t *M, precision x);
void m_elem_add (matrix_t *M, precision x);

/***** 2.1 - returns a matrix, does not change input matrix M *****/
/***** No other inputs, except for m_reshape *****/
// 2.1.0
//	- returns row vector
matrix_t * m_sumCols (matrix_t *M);
matrix_t * m_meanCols (matrix_t *M);
// 2.1.1
//	- returns column vector
matrix_t * m_sumRows (matrix_t *M);
matrix_t * m_meanRows (matrix_t *M);
matrix_t * m_findNonZeros (matrix_t *M);
// 2.1.2
//	- reshapes data in matrix to new form
matrix_t * m_transpose (matrix_t *M);
matrix_t * m_reshape (matrix_t *M, int newNumRows, int newNumCols);

