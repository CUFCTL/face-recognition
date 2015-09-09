#include "matrixOperations5.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 5 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate a multiple matrices but return a matrix of 
 *  inequivalent dimensions.*/
/*******************************************************************************
 * void multiply_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols, int k);
 *
 * product of two matrices (matrix multiplication)
*******************************************************************************/
matrix_t * m_matrix_multiply (matrix_t *A, matrix_t *B, int maxCols) {
	int i, j, k, numCols;
	matrix_t *M;
	numCols = B->numCols;
	if (B->numCols != maxCols && maxCols != 0) {
		printf ("Matrix Size changed somewhere");
		numCols = maxCols;
	}
	M = m_initialize (ZEROS, A->numRows, numCols);
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			for (k = 0; k < A->numCols; k++) {
				elem(M, i, j) += elem(A, i, k) * elem(B, k, j);
			}
		}
	}
	
	return M;
}


/*******************************************************************************
 * void matrix_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows1, int cols1, int rows2, int cols2);
 *
 * multiply one matrix by the inverse of another
*******************************************************************************/
matrix_t * m_matrix_division (matrix_t *A, matrix_t *B) {
	matrix_t *C = m_copy (B);
	m_inverseMatrix (C);
	matrix_t *R = m_matrix_multiply (A, C, 0);
	m_free (C);
	return R;
}


/*******************************************************************************
 * m_reorderCols
 *
 * This reorders the columns of input matrix M to the order specified by V into output matrix R
 *
 * Note:
 * 		V must have 1 row and the number of columns as M
 *
 * ICA:
 * 		void reorder_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t *rowVect);
*******************************************************************************/
matrix_t * m_reorder_columns (matrix_t *M, matrix_t *V) {
	assert (M->numCols == V->numCols && V->numRows == 1);
	
	int i, j, row;
	matrix_t *R = m_initialize (UNDEFINED, M->numRows, M->numCols);
	for (j = 0; j < R->numCols; j++) {
		row = elem(V, 1, j);
		for (i = 0; i < R->numRows; i++) {
			elem(R, i, j) = elem(M, row, j);
		}
	}
	return R;
}



