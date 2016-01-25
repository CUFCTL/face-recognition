#include "matrixOperations.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 2 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
// These operate on input matrix M and will change the data stored in M
//2.0
//2.0.0
/*******************************************************************************
 * m_flipCols
 *
 * Swaps columns in M from left to right 
 *
 * ICA:
 * 		void fliplr(data_t *outmatrix, data_t *matrix, int rows, int cols)
*******************************************************************************/
void m_flipCols (matrix_t *M) {
	int i, j;
	precision temp;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols / 2; j++) {
            temp = elem(M, i, j);
            elem(M, i, j) = elem(M, i, M->numCols - j - 1);
            elem(M, i, M->numCols - j - 1) = temp;
		}
	}
}


/*******************************************************************************
 * void normalize(data_t *outmatrix, data_t *matrix, int rows, int cols); 
 *
 * normalizes the matrix
*******************************************************************************/
void m_normalize (matrix_t *M) {
	int i, j;
	precision max, min, val;
	min = elem(M, 0, 0);
	max = min;
	
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			val = elem(M, i, j);
			if (min > val) {
				min = val;
			}
			if (max < val) {
				max = val;
			}
		}
	}
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			elem(M, i, j) = (elem (M, i, j) - min) / (max - min);
		}
	}
	
}

//2.0.1
/*******************************************************************************
 * m_truncateAll
 *
 * Truncates the entries in matrix M
 *
 * ICA:
 * 		void fix(data_t *outmatrix, data_t *matrix, int rows, int cols);
*******************************************************************************/
void m_elem_truncate (matrix_t *M) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) { 
			elem(M, i, j) = ((precision) ((int)elem(M, i, j)));
		}
	}
}


/*******************************************************************************
 * m_acosAll
 *
 * applies acos to all matrix elements
 
 *
 * ICA:
 * 		 void matrix_acos(data_t *outmatrix, data_t *matrix, int rows, int cols);
*******************************************************************************/
void m_elem_acos (matrix_t *M) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			elem(M, i, j) = acos (elem(M, i, j));
		}
	}
}

/*******************************************************************************
 * void matrix_sqrt(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * applies sqrt to all matrix elements
*******************************************************************************/

void m_elem_sqrt (matrix_t *M) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			elem(M, i, j) = sqrt(elem(M, i, j));
		}
	}
}


/*******************************************************************************
 * void matrix_negate(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * negates all matrix elements
*******************************************************************************/
void m_elem_negate (matrix_t *M) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) = - elem(M, i, j);
		}
	}
}


/*******************************************************************************
 * void matrix_exp(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * raises e to all matrix elements
*******************************************************************************/
void m_elem_exp (matrix_t *M) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) = exp ( elem(M, i, j) );
		}
	}	
}


//2.0.2
/*******************************************************************************
 * void raise_matrix_to_power(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar);
 *
 * raises all matrix elements to a power
*******************************************************************************/
void m_elem_pow (matrix_t *M, precision num) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) = pow ( elem(M, i, j) , num);
		}
	}
}


/*******************************************************************************
 * void scale_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar);
 *
 * scales matrix by contant
*******************************************************************************/
void m_elem_mult (matrix_t *M, precision x) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) =  elem(M, i, j) * x;
		}
	}
}

/*******************************************************************************
 * void divide_by_constant(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar);
 *
 * divides matrix by contant
*******************************************************************************/
void m_elem_divideByConst (matrix_t *M, precision x) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) =  elem(M, i, j) / x;
		}
	}	
}


/*******************************************************************************
 * void divide_scaler_by_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar) ;
 *
 * divides constant by matrix element-wise
*******************************************************************************/
void m_elem_divideByMatrix (matrix_t *M, precision num) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) =  num / elem(M, i, j);
		}
	}	
}


/*******************************************************************************
 * void sum_scalar_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar);
 *
 * adds element-wise matrix to contant
*******************************************************************************/
void m_elem_add (matrix_t *M, precision x) {
	int i, j;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
            elem(M, i, j) =  elem(M, i, j) + x;
		}
	}
}


//2.1
//2.1.0
/*******************************************************************************
 * void sum_columns(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * sums the columns of a matrix, returns a row vector
*******************************************************************************/
matrix_t * m_sumCols (matrix_t *M) {
	matrix_t *R = m_initialize (UNDEFINED, 1, M->numCols);
	int i, j;
	precision val;
	for (j = 0; j < M->numCols; j++) {
		val = 0;
		for (i = 0; i < M->numRows; i++) {
			val += elem(M, i, j);
		}
		elem(R, 0, j) = val;
	}

	return R;
}


/*******************************************************************************
 * void mean_of_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * takes the mean value of each column
*******************************************************************************/
matrix_t *m_meanCols (matrix_t *M) {
	matrix_t *R = m_sumCols (M);
	int i;
	for (i = 0; i < M->numCols; i++) {
		elem(R, 0, i) = elem(R, 0, i) / M->numRows;
    }	
	return R;
}

/*
 * inputs: column vector m will be subtracted from column i
 *         of matrix A
 * outputs: void.  subtraction is done on A
 * note: initially made for PCA
*/
void m_subtractColumn(A,i,m){
    int r;
    int c;
    c = i;
    for(r = 0;r < A->numRows;r++){
        elem(A,r,c) = elem(A,r,c) - elem(m,r,0);
    }
}

//2.1.1
/*******************************************************************************
 * void sum_rows(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * sums the rows of a matrix, returns a col vect
*******************************************************************************/
matrix_t * m_sumRows (matrix_t *M) {
	matrix_t *R = m_initialize (UNDEFINED, M->numRows, 1);
	int i, j;
	precision val;
	for (i = 0; i < M->numRows; i++) {
		val = 0;
		for (j = 0; j < M->numCols; j++) {
			val += elem(M, i, j);
		}
		elem(R, i, 0) = val;
	}

	return R;
}


/*******************************************************************************
 * void mean_of_matrix_by_rows(data_t *outmatrix,data_t *matrix,int rows,int cols);
 *
 * takes the mean of the rows of a matrix, returns a col vect
*******************************************************************************/
matrix_t *m_meanRows (matrix_t *M) {
	matrix_t *R = m_sumRows (M);
	int i;
	for (i = 0; i < M->numRows; i++) {
		elem(R, i, 0) = elem(R, i, 0) / M->numCols;
	}
	
	return R;
}


/*******************************************************************************
 * void find(data_t *outvect, data_t **matrix, int rows, int cols);
 * NOTE: this also assumes that outvect is a column vector)
 * places the row indeces of non-zero elements in a vector
 * This vector has additional, non-used space, not sure what to do about this -miller
*******************************************************************************/
matrix_t * m_findNonZeros (matrix_t *M) {
	matrix_t *R = m_initialize (ZEROS, M->numRows * M->numCols, 1);
	precision val;
	int i, j, count = 0;
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			val = elem(M, i, j);
			if (val != 0) {
				elem(R, count, 0) = (precision) (i + 1);
                count++; 
            }
		}
	}
	return R;
}


//2.1.2
/*******************************************************************************
 * transpose matrix
 *
 * This function transposes a matrix
 *
 * ICA:
 * 		void transpose(data_t *outmatrix, data_t *matrix, int rows, int cols);
*******************************************************************************/
matrix_t * m_transpose (matrix_t *M) {
	int i, j;
	matrix_t *T = m_initialize (UNDEFINED, M->numCols, M->numRows);

	for (i = 0; i < T->numRows; i++) {
		for (j = 0; j < T->numCols; j++) {
			elem(T, i, j) = elem(M, j, i);
		}
	}
	return T;
}

/*******************************************************************************
 * void reshape(data_t **outmatrix, int outRows, int outCols, data_t **matrix, int rows, int cols)
 *
 * reshapes matrix by changing dimensions, keeping data
*******************************************************************************/
matrix_t *m_reshape (matrix_t *M, int newNumRows, int newNumCols) {
	assert (M->numRows * M->numCols == newNumRows * newNumCols);
	int i;
	int r1, c1, r2, c2;
	matrix_t *R = m_initialize (UNDEFINED, newNumRows, newNumCols);
	for (i = 0; i < newNumRows * newNumCols; i++) {
		r1 = i / newNumCols;
		c1 = i % newNumCols;
		r2 = i / M->numCols;
		c2 = i % M->numCols;
		elem(R, r1, c1) = elem(M, r2, c2);
	}
	
	return R;
}




