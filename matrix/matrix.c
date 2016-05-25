/**
 * @file matrix.c
 *
 * Implementation of the matrix library.
 */
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <lapacke.h>
#include "matrix.h"

/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new matrix
 */
matrix_t * m_initialize (int rows, int cols)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->numRows = rows;
	M->numCols = cols;
	M->data = (precision *) malloc(rows * cols * sizeof(precision));

	return M;
}

/**
 * Construct an identity matrix.
 *
 * @param rows
 * @return pointer to a new identity matrix
 */
matrix_t * m_identity (int rows)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->numRows = rows;
	M->numCols = rows;
	M->data = (precision *) malloc(rows * rows * sizeof(precision));

	int i;
	for ( i = 0; i < rows; i++ ) {
		elem(M, i, i) = 1;
	}

	return M;
}

/**
 * Construct a zero matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new zero matrix
 */
matrix_t * m_zeros (int rows, int cols)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->numRows = rows;
	M->numCols = cols;
	M->data = (precision *) calloc(rows * cols, sizeof(precision));

	return M;
}

/**
 * Copy a matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to copy of M
 */
matrix_t * m_copy (matrix_t *M)
{
	matrix_t *C = m_initialize(M->numRows, M->numCols);

	memcpy(C->data, M->data, C->numRows * C->numCols * sizeof(precision));

	return C;
}

/**
 * Deconstruct a matrix.
 *
 * @param M  pointer to matrix
 */
void m_free (matrix_t *M)
{
	free(M->data);
	free(M);
}

/**
 * Write a matrix in text format to a stream.
 *
 * @param stream  pointer to file stream
 * @param M       pointer to matrix
 */
void m_fprint (FILE *stream, matrix_t *M)
{
	fprintf(stream, "%d %d\n", M->numRows, M->numCols);

	int i, j;
	for ( i = 0; i < M->numRows; i++ ) {
		for ( j = 0; j < M->numCols; j++ ) {
			fprintf(stream, "%lf ", elem(M, i, j));
		}
		fprintf(stream, "\n");
	}
}

/**
 * Write a matrix in binary format to a stream.
 *
 * @param stream  pointer to file stream
 * @param M       pointer to matrix
 */
void m_fwrite (FILE *stream, matrix_t *M)
{
	fwrite(&M->numRows, sizeof(int), 1, stream);
	fwrite(&M->numCols, sizeof(int), 1, stream);
	fwrite(M->data, sizeof(precision), M->numRows * M->numCols, stream);
}

/**
 * Read a matrix in text format from a stream.
 *
 * @param stream  pointer to file stream
 * @return pointer to new matrix
 */
matrix_t * m_fscan (FILE *stream)
{
	int numRows, numCols;
	fscanf(stream, "%d %d", &numRows, &numCols);

	matrix_t *M = m_initialize(numRows, numCols);
	int i, j;
	for ( i = 0; i < numRows; i++ ) {
		for ( j = 0; j < numCols; j++ ) {
			fscanf(stream, "%lf", &(elem(M, i, j)));
		}
	}

	return M;
}

/**
 * Read a matrix in binary format from a stream.
 *
 * @param stream  pointer to file stream
 * @return pointer to new matrix
 */
matrix_t * m_fread (FILE *stream)
{
	int numRows, numCols;
	fread(&numRows, sizeof(int), 1, stream);
	fread(&numCols, sizeof(int), 1, stream);

	matrix_t *M = m_initialize(numRows, numCols);
	fread(M->data, sizeof(precision), M->numRows * M->numCols, stream);

	return M;
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
	matrix_t *R = m_initialize(1, M->numCols);
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
void m_subtractColumn(matrix_t *M,int i,matrix_t *m){
    int r;
    int c;
    c = i;
    for(r = 0;r < M->numRows;r++){
        M->data[c * M->numRows + r] -= m->data[r];
        //elem(M,r,c) = elem(M,r,c) - elem(m,r,0);
    }
}

//2.1.1
/*******************************************************************************
 * void sum_rows(data_t *outmatrix, data_t *matrix, int rows, int cols);
 *
 * sums the rows of a matrix, returns a col vect
*******************************************************************************/
matrix_t * m_sumRows (matrix_t *M) {
	matrix_t *R = m_initialize(M->numRows, 1);
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
	matrix_t *R = m_zeros(M->numRows * M->numCols, 1);
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
	matrix_t *T = m_initialize(M->numCols, M->numRows);

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
	matrix_t *R = m_initialize(newNumRows, newNumCols);
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
// Temp moved this here for dependancy issues. Group 3 code person should work on this
// for right now.
// TODO - Documentation & Free's - Miller 10/30
// UPDATE 03/02/16: make sure to compile time include -llapacke
// NEEDS VERIFICATION - Greg
void m_inverseMatrix (matrix_t *M) {
    //assert(M->numRows == M->numCols);
    int info;
    //int lwork = M->numRows * M->numCols;
    int *ipiv = malloc((M->numRows + 1) * sizeof(int));
    //precision *work = malloc(lwork * sizeof(precision));
    //          (rows   , columns, matrix , lda    , ipiv, info );
    info=LAPACKE_dgetrf(LAPACK_ROW_MAJOR,M->numCols, M->numRows, M->data, M->numRows, ipiv);
    if(info!=0){
        //printf("\nINFO != 0\n");
        exit(1);
    }
    //printf("\ndgertrf passed\n");
    //          (order  , matrix, Leading Dim, IPIV,
    info=LAPACKE_dgetri(LAPACK_ROW_MAJOR,M->numCols,M->data,M->numRows, ipiv);
    if(info!=0){
        //printf("\nINFO2 != 0\n");
        exit(1);
    }
}

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
		A = m_initialize(M->numRows - 1, M->numCols - 1);
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
    matrix_t *A = m_initialize(M->numRows - 1, M->numCols - 1);
	matrix_t *R = m_initialize(M->numRows, M->numCols);
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
matrix_t * m_covariance(matrix_t *M) {
	int i, j, k;
	precision val;
	matrix_t *colAvgs = m_meanCols(M);
	matrix_t *norm = m_initialize(M->numRows, M->numCols);
	matrix_t *R = m_initialize(M->numRows, M->numCols);

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
 *   *  equivalent dimensions.
 *    */
/*******************************************************************************
 *  * void subtract_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
 *   *
 *    * return the difference of two matrices element-wise
 *    *******************************************************************************/
matrix_t * m_dot_subtract (matrix_t *A, matrix_t *B) {
    assert (A->numRows == B->numRows && A->numCols == B->numCols);
    matrix_t *R = m_initialize(A->numRows, A->numCols);
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
    matrix_t *R = m_initialize(A->numRows, A->numCols);
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
    matrix_t *R = m_initialize(A->numRows, A->numCols);
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
 * TODO these functions should not include maxcols as an argument
*******************************************************************************/
//matrix_t * m_matrix_multiply (matrix_t *A, matrix_t *B, int maxCols) {
matrix_t * m_matrix_multiply (matrix_t *A, matrix_t *B){
	int i, j, k, numCols;
	matrix_t *M;
	numCols = B->numCols;
	/*if (B->numCols != maxCols && maxCols != 0) {
		printf ("Matrix Size changed somewhere");
		numCols = maxCols;
	}*/
	M = m_zeros(A->numRows, numCols);
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
	matrix_t *R = m_matrix_multiply (A, C);
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
	matrix_t *R = m_initialize(M->numRows, M->numCols);
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
 * TODO Change to allow for submatrix maybe
 * TODO Change to allow for passing in of work space?
*******************************************************************************/
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors) {
    // Right eigenvector of (A,B)
    // Satisfies A * v = \lambda * B * v
    // In this case, B = IDENT

    assert(M->numCols == M->numRows);

    int N = M->numCols; // Order of all matrices involved
    int LDA = M->numCols; // Leading dimension of A
    int LDVR = M->numCols;
    int LWORK = M->numCols;
    int INFO;
    precision* A = M->data;

    // Output
    precision *ALPHAR = (precision *)calloc(N, sizeof(precision)); // Real components
    precision *ALPHAI = (precision *)calloc(N, sizeof(precision)); // Imaginary components
    precision *BETA = (precision *)calloc(N, sizeof(precision));   // Scale
    precision *VR = (precision *)calloc(LDVR, sizeof(precision));  // Eigenvectors
    precision *WORK = (precision *)calloc(LWORK, sizeof(precision));

    // ALPHAR on completion is normalized eigenvalues diagonal
    // VR is right eigenvectors

	// TODO: compiler says 'undefined reference to dggev'
    // dggev('N', 'V', N, A, LDA, (NULL), 1, 1, 1, ALPHAR, ALPHAI, BETA,
    //         (NULL), 1, VR, LDVR, WORK, LWORK, &INFO);

    // Free memory
    free(ALPHAI);
    free(BETA);
    free(WORK);

    // Return values
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
void loadPPMtoMatrixCol (const char *filename, matrix_t *M, int specCol, unsigned char *pixels) {
	FILE *in = fopen (filename, "r");
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
void writePPMgrayscale (const char *filename, matrix_t *M, int specCol, int height, int width) {

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
