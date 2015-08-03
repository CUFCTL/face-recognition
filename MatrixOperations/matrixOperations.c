#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_eigen.h>
#include "matrixOperations.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 1 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
//  initialization, free, input, output, and copy functions
/*******************************************************************************
 * m_initialize
 *
 * Returns a matrix pointer to a matrix of size M x N
 *
 * Depending on the input variable "mode", data is either a 2D matrix of
 * 		1. ZEROS = zeros
 *		2. IDENTITY = identity matrix
 *		3. UNDEFINED = undefined values
 * 		4. ONES = all ones
 *		5. FILL = each element increases by one
 *
 * ICA:
 *		void allocate_matrix(data_t **vector, int rows, int cols);
 *		void allocate_vector(data_t **vector, int length);
 *		void ones(data_t *onesMat, int rows, int cols);
 * 		void eye(data_t *identity, int dimension);
 * 		void fill_matrix(data_t *matrix, int rows, int cols);
*******************************************************************************/
matrix_t * m_initialize (int mode, int numRows, int numCols) {
	int i, j;
	matrix_t *M = (matrix_t *) malloc (sizeof (matrix_t));
	M->numRows = numRows;
	M->numCols = numCols;
	M->span = numCols;
	M->type = PARENT; // not submatrix
	if (mode == ZEROS || mode == IDENTITY) {
		M->data = (precision *) calloc (numRows * numCols, sizeof (precision));
		if (mode == IDENTITY) {
			assert (numRows == numCols);
			for (i = 0; i < numRows; i++) {
                elem(M, i, i) = 1;	
			}
		}
	} else if (mode == UNDEFINED || mode == ONES || mode == FILL){
		M->data = (precision *) malloc (numRows * numCols * sizeof (precision));
		if (mode == ONES) {
			for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
				    elem(M, i, j) = 1.0;
                }
			}
		} else if (mode == FILL) {
			for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
				    elem(M, i, j) = i * numRows + j;
                }
			}
		}
	} else {
		printf ("m_initialize, Not valid mode\n");
		exit (5);
	}
	
	return M;
}


/*******************************************************************************
 * m_free
 * 
 * Frees memory for matrix M
 * ICA:
 *		void free_matrix(data_t **matrix);
 *		void free_vector(data_t **vector);
*******************************************************************************/
void m_free (matrix_t *M) {
	if (M->type != SUBMATRIX) {
		free (M->data);
	}
	free (M);
}


/*******************************************************************************
 * m_fprint
 * 
 * Prints matrix M to the stream specified
 * Prints numRows, numCols, then each whole row of the matrix (aka [0][0], [0][1]..)
 *
 * ICA:
 *		void print_matrix(data_t *matrix, int rows, int cols);
*******************************************************************************/
void m_fprint (FILE *stream, matrix_t *M) {

	int i, j;
	
	fprintf (stream, "%d %d\n", M->numRows, M->numCols);
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			fprintf (stream, "%lf ", elem(M, i, j));
		}
		fprintf (stream, "\n");
	}
	fflush (stream);
}


/*******************************************************************************
 * m_fwrite
 * 
 * Writes matrix M to the stream specified
 * Writes numRows, numCols, then the data
 * NOTE: will not work with submatrixes right now
*******************************************************************************/
void m_fwrite (FILE *stream, matrix_t *M) {
	fwrite (&M->numRows, sizeof (unsigned long int), 1, stream);
	fwrite (&M->numCols, sizeof (unsigned long int), 1, stream);
	if (M->type == PARENT) {
		fwrite (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	} else {
		int i;
		for (i = 0; M->numRows; i++) {
			fwrite (&(M->data[i * M->span]), sizeof (precision), M->numCols, stream);
		}
	}
}


/*******************************************************************************
 * m_fscan
 *
 * Scans matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * m_fscan (FILE *stream) {

	int i, j, numRows, numCols;
	numRows = 0;
	numCols = 0;	

	fscanf (stream, "%d %d", &numRows, &numCols);
	matrix_t *M = m_initialize(UNDEFINED, numRows, numCols);
	for (i = 0; i < numRows; i++) {
		for (j = 0; j < numCols; j++) {
			fscanf (stream, "%lf", &(elem(M, i, j)));
		}
	}

	return M;
}

/*******************************************************************************
 * m_fread
 *
 * reads matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * m_fread (FILE *stream) {
	int numRows, numCols;
	fread (&numRows, sizeof (unsigned long int), 1, stream);
	fread (&numCols, sizeof (unsigned long int), 1, stream);
	matrix_t *M = m_initialize (UNDEFINED, numRows, numCols);
	fread (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	return M;
}


/*******************************************************************************
 * m_copy
 *
 * Copies matrix M into a new matrix
 *
 * ICA:
 * 		data_t* copy(data_t* orig,int rows,int cols);
*******************************************************************************/
matrix_t * m_copy (matrix_t *M) {
	int i, j;
	
	matrix_t *C = (matrix_t *) malloc (sizeof(matrix_t));
	C->numRows = M->numRows;
	C->numCols = M->numCols;
	
	C->data = (precision *) malloc (C->numRows * C->numCols * sizeof (precision));
	if (M->numCols == M->span) {
		memcpy(C->data, M->data, C->numRows * C->numCols * sizeof (precision));
	} else {
		for (i = 0; i < C->numRows; i++) {
			for (j = 0; j < C->numCols; j++) {
                elem(C, i, j) = elem(M, i, j);
            }
		}
	}
	return C;
}


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


/*******************************************************************************
 * void inv(data_t *outmatrix, data_t *matrix, int rows);
 *
 * inverse of the matrix
*******************************************************************************/
void m_inverseMatrix (matrix_t *M) {
	
	matrix_t *cofactorMatrix = m_cofactor (M);
	precision det = m_determinant (M);
	m_elem_divideByConst (cofactorMatrix, det);
	
    m_free(M);
    M = m_copy(cofactorMatrix);

	m_free (cofactorMatrix);
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
	
	matrix_t *eigenvectors;
	matrix_t *eigenvalues;
	m_eigenvalues_eigenvectors (M, &eigenvalues, &eigenvectors);
	
	m_elem_sqrt (eigenvalues);
	
	matrix_t * temp = m_matrix_multiply (eigenvectors, eigenvalues, 0);
	m_free (eigenvalues);
	matrix_t * R = m_matrix_division (temp, eigenvectors);
	m_free (temp);
	m_free(eigenvectors);
	return R;
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
			sign = 2 * (2 % (j + 1)) - 1;
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
			if (j % 2 == 0) {
                if (i % 2 == 0)
                    sign = 1;
                else
                    sign = -1;
            } else {
                if (i % 2 == 0)
                    sign = -1;
                else
                    sign = 1;
            }
			val *=sign;
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


/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 4 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate multiple matrices and return a matrix of 
 *  equivalent dimensions.  
 */
/*******************************************************************************
 * void subtract_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *
 * return the difference of two matrices element-wise
*******************************************************************************/
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
 * void add_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *
 * element wise sum of two matrices
*******************************************************************************/
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
 * void matrix_dot_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *
 * element wise division of two matrices
*******************************************************************************/
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


/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 6 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*******************************************************************************
 * void matrix_eig(data_t *out_eig_vect, data_t*out_eig_vals, data_t* matrix, int rows, int cols); 
 * Get eigenvalues and eigenvectors of symmetric matrix
 * NOTE: ONLY SYMMETRIC MATRICIES ATM
*******************************************************************************/
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors) {
	/*gsl_matrix * A = gsl_matrix_alloc (M->numRows, M->numCols);
	gsl_matrix * gslEigenvectors = gsl_matrix_alloc (M->numRows, M->numCols);
	gsl_vector * gslEigenvalues = gsl_vector_alloc (M->numRows);
	
	precision val;
	int i, j;
	// Copy M into A
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			val = m_getelem(M, i, j);
			gsl_matrix_set (A, i, j, val);
		}
	}

	// Compute the Eigenvalues using the GSL library
	// Allocate workspace
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (M->numRows);

	gsl_eigen_symmv (A, gslEigenvalues, gslEigenvectors, w);

	// ********************************************************
	// COMMENT
	// We might need to normalize the eigenvectors here or something
	// to match matlab eigenvectors, they don't HAVE to to match but
	// its at least something to keep in mind
	// ********************************************************
	
	matrix_t *eigenvalues = m_initialize (UNDEFINED, gslEigenvalues->size, 1);
	matrix_t *eigenvectors = m_initialize (UNDEFINED, gslEigenvectors->size1, gslEigenvectors->size2);

	// Copy the eigenvalues into a column matrix
	for (i = 0; i < gslEigenvalues->size; i++) {
		val = gsl_vector_get (gslEigenvalues, i);
		m_setelem(val, eigenvalues, i, 0);
	}
	
	// Copy the eigenvectors into a regular matrix
	for (i = 0; i < gslEigenvectors->size1; i++) {
		for (j = 0; j < gslEigenvectors->size2; j++) {
			val = gsl_matrix_get (gslEigenvectors, i, j);
			m_setelem(val, eigenvectors, i, j);
		}
	}
	gsl_eigen_symmv_free (w);
	gsl_matrix_free (gslEigenvectors);
	gsl_matrix_free (A);
	gsl_vector_free (gslEigenvalues);
	
	*p_eigenvectors = eigenvectors;
	*p_eigenvalues = eigenvalues; */
    
    *p_eigenvalues = m_initialize (ZEROS, M->numRows, 1);
    *p_eigenvectors = m_initialize (ZEROS, M->numRows, M->numCols);

}


/*******************************************************************************
 * void submatrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int start_row, int start_col, int end_row, int end_col);
 * NOTE: THIS DIRECTLY MANIPULATES THE PARENTS DATA
*******************************************************************************/
matrix_t * m_getSubMatrix (matrix_t *M, int startRow, int startCol, int numRows, int numCols) {
	matrix_t *sub = (matrix_t *) malloc (sizeof (matrix_t));
	sub->numRows = numRows;
	sub->numCols = numCols;
	sub->span = M->span;
	sub->type = SUBMATRIX;
	sub->data = &(M->data[numRows * M->span + numCols]);

	return sub;
}


/*******************************************************************************
 * Helper functio just used for function below
*******************************************************************************/
void skip_to_next_value(FILE* in)
{
   char ch = fgetc(in);
   while(ch == '#' || isspace(ch))
   {
       if(ch == '#')
       {
          while(ch != '\n') 
          {
             ch = fgetc(in);
          }      
       }
       else
       {
          while(isspace(ch))
          {
             ch = fgetc(in);             
          }
       }
   }
   
   ungetc(ch,in); //return last read value
}

/*******************************************************************************
 * loadPPMtoMatrixCol
 * 
 * This function loads the pixel data of a PPM image as a single column vector
 * in the preinitialized matrix M. It will load it into the column specified as
 * the specCol parameter. 
 * 
 * This function automatically turns any picture to grayscale if it is not
 * already
 * NOTE : currently this is set manually with the #define IS_COLOR in matrix.h
 *
 * NOTE : pixels is a matrix that must be allocated beforehand. This is to speed
 * up execution time if this function is called multiple times on the same size
 * image as it doesn't have to malloc and free that array every time.
*******************************************************************************/
void loadPPMtoMatrixCol (char *path, matrix_t *M, int specCol, unsigned char *pixels) {
	FILE *in = fopen (path, "r");
	char header[4];
	int height, width, size, i;
	int numPixels = M->numRows;
	precision intensity;
	
	fscanf (in, "%s", header);
	if (strcmp (header, "P3") == 0) {
		skip_to_next_value (in);
		fscanf (in, "%d", &height);
		skip_to_next_value (in);
		fscanf (in, "%d", &width);
		skip_to_next_value (in);
		fscanf (in, "%d", &size);
		skip_to_next_value (in);
		for (i = 0; i < numPixels * 3; i++) {
			fscanf(in, "%c", &pixels[i]);
		}
	} else if (strcmp (header, "P6") == 0){
		fscanf (in, "%d %d %d", &height, &width, &size);
		skip_to_next_value(in);
		fread (pixels, 3 * sizeof (unsigned char), numPixels, in);
	} else {
		printf ("Error not a P3 or P6 PPM");
		exit (8);
	}

	for (i = 0; i < numPixels; i++) {
		intensity = 0.299 * (precision)pixels[3*i] +
					0.587 * (precision)pixels[3*i+1] +
					0.114 * (precision) pixels[3*i+2];
		elem(M, i, specCol) = intensity;
	}
	
	fclose (in);
}


/*******************************************************************************
 * writePPMgrayscale
 *
 * This writes a column vector of M (column specified by specCol) as a
 * grayscale ppm image. The height and width of the image must be specified 
 *
*******************************************************************************/
void writePPMgrayscale (char * filename, matrix_t *M, int specCol, int height, int width) {

	int i;
	char c;

	assert (height * width == M->numRows); // Number of pixels must match
	FILE * out = fopen (filename, "w");

	// Write file header
	fprintf (out, "P6\n%d\n%d\n255\n", height, width);

	// Write pixel data
	for (i = 0; i < M->numRows; i++) {
		c = (char) elem(M, i, specCol);
		fputc (c, out);
		fputc (c, out);
		fputc (c, out);
	}
	fclose (out);
}

