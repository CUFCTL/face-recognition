#include "matrixOperations3.h"


// Temp moved this here for dependancy issues. Group 3 code person should work on this for right now. 
void m_inverseMatrix (matrix_t *M) {
	
	matrix_t *cofactorMatrix = m_cofactor (M);
	matrix_t *transpose = m_transpose (cofactorMatrix);
	precision det = m_determinant (M);
    m_elem_divideByConst (transpose, det);
	
    m_free (M);
    M = m_copy(transpose);

	m_free (transpose);
    m_free (cofactorMatrix);
}




/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 3 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*******************************************************************************
 * data_t norm(data_t *matrix, int rows, int cols);
 *
 * NOTE: I think the norm def I looked up was different. I kept this one for 
 * right now but we need to look at that again
*******************************************************************************/
precision m_norm (matrix_t *M, int specRow) {
	int i, j;
	precision val, sum = 0;
	
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			val = elem(M, i, j);
			sum += val * val;
		}
	}
	
	return sqrt (sum);
}


/*******************************************************************************
 * void matrix_sqrtm(data_t *outmatrix, data_t *matrix, int rows, int cols);
 * 
 * matrix square root
 *  element-wise square rooting of eigenvalues matrix
 *  divide eigenvectors matrix by the product of the e-vectors and e-values
*******************************************************************************/
matrix_t * m_sqrtm (matrix_t *M) {
	/*
	matrix_t *eigenvectors;
	matrix_t *eigenvalues;
    // TODO: EIGENVALUES NOT CURRENTLY WORKING
	m_eigenvalues_eigenvectors (M, &eigenvalues, &eigenvectors);
	
	m_elem_sqrt (eigenvalues);
	
	matrix_t * temp = m_matrix_multiply (eigenvectors, eigenvalues, 0);
	m_free (eigenvalues);
	matrix_t * R = m_matrix_division (temp, eigenvectors);
	m_free (temp);
	m_free(eigenvectors);
	return R;*/
    return M;
}


/*******************************************************************************
 * void determinant(data_t *matrix, int rows, double *determ);
 *
 * find the determinant of the matrix
*******************************************************************************/
precision m_determinant (matrix_t *M) {
	//int i, j, j1, j2;
	int i, j, r, c, k, sign;
    precision det = 0, val;
    matrix_t *A = NULL;
	assert (M->numCols == M->numRows);
	
    if (M->numRows < 1)   printf("error finding determinant\n");
    else if (M->numRows == 1) det = elem(M, 0, 0); // Shouldn't get used
    else if (M->numRows == 2) det = elem(M, 0, 0) * elem(M, 1, 1) - elem(M, 1, 0) * elem(M, 0, 1);
    else {
        
		det = 0;
		A = m_initialize (UNDEFINED, M->numRows - 1, M->numCols - 1);
		for (j = 0; j < M->numCols; j++) {
			// Fill matrix
			c = 0;
			for (k = 0; k < M->numCols; k++) {
				if (k == j) continue; // skip over columns that are the same
				for (i = 1; i < M->numRows; i++) {
					r = i - 1;
					elem(A, r, c) = elem(M, i, k);
				}
				c++;
			}
			val = m_determinant (A);
			sign = 1 - 2 * ((i % 2) ^ (j % 2));;
			det += sign * elem(M, 0, j) * val;
		}
		m_free (A);
						
    }
	return det;
}


/*******************************************************************************
 * void cofactor(data_t *outmatrix,data_t *matrix, int rows);
 *
 * cofactor a matrix
*******************************************************************************/
matrix_t * m_cofactor (matrix_t *M) {
	//int i, j, ii, jj, i1, j1;
	int i, j, r, c, row, col, sign;
	assert (M->numRows == M->numCols);
    matrix_t *A = m_initialize (UNDEFINED, M->numRows - 1, M->numCols - 1);
	matrix_t *R = m_initialize (UNDEFINED, M->numRows, M->numCols);
	precision val;
	
	// For every element in M
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			// Make matrix of values not sharing this column/row
			for (r = 0, row = 0; r < M->numRows; r++) {
				if (i == r) continue;
				for (c = 0, col = 0; c < M->numCols; c++) {
					if (j == c) continue;
					elem(A, row, col) = elem(M, r, c);
					col++;
				}
				row++;
			}
			val = m_determinant (A);
			sign = 1 - 2 * ((i % 2) ^ (j % 2)); // I think this is illegal
            val *= sign;
			elem(R, j, i) = val;
		}
	}
	m_free (A);
	return R;
}


/*******************************************************************************
 * void covariance(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * return the covariance matrix
*******************************************************************************/
matrix_t * m_covariance (matrix_t *M) {
	int i, j, k;
	precision val;
	matrix_t *colAvgs = m_meanCols(M);
	matrix_t *norm = m_initialize (UNDEFINED, M->numRows, M->numCols);
	matrix_t *R = m_initialize (UNDEFINED, M->numRows, M->numCols);
	
	for (j = 0; j < M->numCols; j++) {
		for (i = 0; i < M->numRows; i++) {
			val = elem(M, i, j) - elem(colAvgs, 0, j);
		}
	}
	
	for (j = 0; j < M->numCols; j++) {
		for (k = 0; k < M->numCols; k++) {
			val = 0;
			for (i = 0; i < M->numRows; i++) {
				val += elem(norm, i, j) * elem(norm, i, j);
			}
			val /= M->numCols - 1;
			elem(R, j, k) = val;
		}
	}
	
	m_free (colAvgs);
	m_free (norm);
	
	return R;
}


