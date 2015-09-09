#include "matrixOperations4.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 4 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate multiple matrices and return a matrix of 
 *   *  equivalent dimensions.  
 *    */
/*******************************************************************************
 *  * void subtract_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *   *
 *    * return the difference of two matrices element-wise
 *    *******************************************************************************/
matrix_t * m_dot_subtract (matrix_t *A, matrix_t *B) {
    assert (A->numRows == B->numRows && A->numCols == B->numCols);
    matrix_t *R = m_initialize (UNDEFINED, A->numRows, A->numCols);
    int i, j;
    for (i = 0; i < A->numRows; i++) {
        for (j = 0; j < A->numCols; j++) {
            elem(R, i, j) = elem(A, i, j) - elem(B, i, j);
        }
    }
    return R;
}


/*******************************************************************************
 *  * void add_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *   *
 *    * element wise sum of two matrices
 *    *******************************************************************************/
matrix_t * m_dot_add (matrix_t *A, matrix_t *B) {
    assert (A->numRows == B->numRows && A->numCols == B->numCols);
    matrix_t *R = m_initialize (UNDEFINED, A->numRows, A->numCols);
    int i, j;
    for (i = 0; i < A->numRows; i++) {
        for (j = 0; j < A->numCols; j++) {
            elem(R, i, j) = elem(A, i, j) + elem(B, i, j);
        }
    }
    return R;
}


/*******************************************************************************
 *  * void matrix_dot_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *   *
 *    * element wise division of two matrices
 *    *******************************************************************************/
matrix_t * m_dot_division (matrix_t *A, matrix_t *B) {
    assert (A->numRows == B->numRows && A->numCols == B->numCols);
    matrix_t *R = m_initialize (UNDEFINED, A->numRows, A->numCols);
    int i, j;
    for (i = 0; i < A->numRows; i++) {
        for (j = 0; j < A->numCols; j++) {
            elem(R, i, j) = elem(A, i, j) / elem(B, i, j);
        }
    }
    return R;
}
