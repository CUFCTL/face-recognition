/*==================================================================================================
 *  matrix_manipulation.c
 *
 *  Edited by William Halsey, Scott Rodgers, Isaac Roberts
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *  irobert@g.clemson.edu
 *
 *  This file contains
 *		allocate_matrix
 *      allocate_vector
 *      free_matrix
 *      free_vector
 *      print_matrix
 *      ones
 *      eye
 *      fill_matrix
 *      copy
 *
 *      fix
 *      fliplr
 *      reorder_matrix
 *      matrix_acos
 *      raise_matrix_to_power
 *      scale_matrix
 *      divide_by_constant
 *      divide_scaler_by_matrix
 *      sum_scalar_matrix
 *      normalize
 *      matrix_sqrt
 *      matrix_sqrtm
 *      matrix_eig
 *      matrix_negate
 *		matrix_exp
 *
 *
 *      transpose
 *      mean_of_matrix
 *      mean_of_matrix_by_rows
 *      find
 *      sum_rows
 *      sum_columns
 *      norm
 *		determinant
 *      inv
 *      cofactor
 *      covariance
 *
 *      subtract_matrices
 *		matrix_dot_division
 *      matrix_division
 *
 *      multiply_matrices
 *
 *  Lasted Edited: Jul. 24, 2013
 *
 *  Changes made: by William - added the #ifndef statement that will resolve any issues involving
 *  including this library in multiple locations.
 *
 * 
 *  by Isaac - added mean_of_matrix_by_rows to replace a call for mean_of_matrix(transpose(x))
 *			 - added copy function
 */
#ifndef __matrix_manipulation_h__
#define __matrix_manipulation_h__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include "/Users/geddingsbarrineau/Documents/lapack-3.5.0/lapacke/include/lapacke.h" 
/* Only for Geddings' OSX*/


typedef double data_t;

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 1 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions do not do any matrix manipulation. They only initialize matrix pointers and   */
/*  matrix values.                                                                                */
void allocate_matrix(data_t **vector, int rows, int cols);
void allocate_vector(data_t **vector, int length);
void free_matrix(data_t **matrix);
void free_vector(data_t **vector);
void print_matrix(data_t *matrix, int rows, int cols);
void ones(data_t *onesMat, int rows, int cols);
void eye(data_t *identity, int dimension);
void fill_matrix(data_t *matrix, int rows, int cols);
data_t* copy(data_t* orig,int rows,int cols);

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 2 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate a single matrix and return a matrix of equivalent dimensions.      */
void fix(data_t *outmatrix, data_t *matrix, int rows, int cols);
void fliplr(data_t *outmatrix, data_t *matrix, int rows, int cols);
void reorder_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t *rowVect);
void matrix_acos(data_t *outmatrix, data_t *matrix, int rows, int cols);
void raise_matrix_to_power(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar);
void scale_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar);
void divide_by_constant(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar);
void divide_scaler_by_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar) ;
void sum_scalar_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar);
void normalize(data_t *outmatrix, data_t *matrix, int rows, int cols);
void matrix_sqrt(data_t *outmatrix, data_t *matrix, int rows, int cols);
void matrix_sqrtm(data_t *outmatrix, data_t *matrix, int rows, int cols);
void matrix_eig(data_t *out_eig_vect, data_t*out_eig_vals, data_t* matrix, int rows, int cols); 
void matrix_negate(data_t *outmatrix, data_t *matrix, int rows, int cols);
void matrix_exp(data_t *outmatrix, data_t *matrix, int rows, int cols);

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 3 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate a single matrix but return a matrix of inequivalent dimensions.    */
void transpose(data_t *outmatrix, data_t *matrix, int rows, int cols);
void mean_of_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols); 
void mean_of_matrix_by_rows(data_t *outmatrix,data_t *matrix,int rows,int cols);
void find(data_t *outvect, data_t **matrix, int rows, int cols);
void sum_rows(data_t *outmatrix, data_t *matrix, int rows, int cols);
void sum_columns(data_t *outmatrix, data_t *matrix, int rows, int cols);
void reshape(data_t **outmatrix, int outRows, int outCols, data_t **matrix, int rows, int cols);
data_t norm(data_t *matrix, int rows, int cols);
void sum_matrix_along_columns(data_t **matrix, data_t *vector, int rows, int cols);
void inv(data_t *outmatrix, data_t *matrix, int rows);
void cofactor(data_t *outmatrix,data_t *matrix, int rows);
void determinant(data_t *matrix, int rows, double *determ);
void covariance(data_t *outmatrix, data_t *matrix, int rows, int cols);
void submatrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int start_row, int start_col, int end_row, int end_col);
/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 4 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate multiple matrices and return a matrix of equivalent dimensions.  */
void subtract_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols); 
void add_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
void matrix_dot_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols);
void matrix_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows1, int cols1, int rows2, int cols2);

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 5 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*  These functions manipulate a multiple matrices but return a matrix of inequivalent dimensions.*/
void multiply_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols, 
    int k);

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 1 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */

/*==================================================================================================
 *  allocate_matrix
 *
 *  Parameters
 *      triple pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a pointer to a matrix through variable "matrix."
 *
 *  Description: This function takes the inputs, rows and cols, and returns a data_t pointer to a
 *  newly dynamically allocated matrix. The dimensions of the matrix will be rows X cols.
 *
 */
void allocate_matrix(data_t **matrix, int rows, int cols) {
    *matrix = (data_t *)malloc(rows * cols * sizeof(data_t));

    return;
}

/*==================================================================================================
 *  allocate_vector
 *
 *  Parameters
 *      double pointer, type data_t = vector
 *      value, type integer         = length
 *
 *  Returns
 *      N/A
 *      Implicitly returns a pointer to a matrix through variable "vector."
 *
 *  Description: This function takes the inputs, rows and cols, and returns a data_t pointer to a
 *  newly dynamically allocated matrix. The dimensions of the matrix will be rows X cols.
 *
 */
void allocate_vector(data_t **vector, int length) {
    
    *vector = (data_t *)malloc(length * sizeof(data_t));

    return;
}

/*==================================================================================================
 *  free_matrix
 *
 *  Parameters
 *      triple pointer, type data_t = matrix
 *      value, type integer         = rows
 *
 *  Returns
 *      N/A
 *
 *  Description: This function frees the memory of a two dimensional array by freeing all of the 
 *  rows of the array and then the pointer to the matrix, in that order.
 */
void free_matrix(data_t **matrix) {
    free(*matrix);
    
    return;
}

/*==================================================================================================
 *  free_vector
 *
 *  Parameters
 *      double pointer, type data_t = vector
 *
 *  Returns
 *      N/A
 *
 *  Description: This function frees the memory of a two dimensional array by freeing all of the 
 *  rows of the array and then the pointer to the matrix, in that order.
 *
 */
void free_vector(data_t **vector) {

    free(*vector);
    
    return;
}

/*==================================================================================================
 *  print_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *
 *  Description: Prints a matrix to the screen
 *
 */
void print_matrix(data_t *matrix, int rows, int cols) {
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)  printf("%10.3e ", matrix[i * cols + j]);
        printf("\n");
    }
    
    return;
}

/*==================================================================================================
 *  ones
 *
 *  Parameters
 *      double pointer, type data_t = onesMat
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a vector through variable 'onesMat.'
 *
 *  Description: Fills a matrix with dimensions rows X cols with 1's.
 *      [1 1 ... 1]
 *      [1 1 ... 1]
 *       : :     :
 *      [1 1 ... 1]
 *
 */
void ones(data_t *onesMat, int rows, int cols) {
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)  onesMat[i * cols + j] = 1;
    }
    
    return;
}

/*==================================================================================================
 *  eye
 *
 *  Parameters
 *      double pointer, type data_t = identity
 *      value, type integer         = dimension
 *
 *  Returns
 *      N/A
 *      implicitly returns a matrix through variable "identity"
 *
 *  Description: Returns a square identity matrix with size dimension X dimension
 *      [1 0 ... 0]
 *      [0 1 ... 0]
 *       : :     :
 *      [0 0 ... 1]
 *
 */
void eye(data_t *identity, int dimension) {
    int i, j;
    
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            if (i == j) identity[i * dimension + j] = 1;
            else    identity[i * dimension + j] = 0;
        }
    }
    return;
}
 
 /*==================================================================================================
 *  fill_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *		value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      implicitly returns a matrix through variable "matrix"
 *
 *  Description: Returns a matrix with each element containing its one base index as the value.
 *      [  0        1    ...  j-1]
 *      [  j       j+1   ... 2j-1]
 *         :        :          :
 *      [(i-1)j (i-1)j+1 ... ij-1]  where i = rows and j = cols
 *
 */
void fill_matrix(data_t *matrix, int rows, int cols){
int i, j;
data_t cnt = 0;

	for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {	
		    matrix[i * cols + j] = cnt; 
		    cnt++;
		}
	}
    
    return;
}

/*========================================================
*  copy
*   Parameters
* 		double pointer(matrix) original
*		int rowAmt
*		int columnAmt
*
*	returns a newly allocated matrix with the same values and size as the original
*	

*	Description: allocates and fills the returned matrix with the same values as the matrix passed in.
*			The matrix passed in is not edited.
*			
*/
data_t* copy(data_t* orig,int rows,int cols){
	data_t* copyTo;
	allocate_matrix(&copyTo,rows,cols);
	int size=rows*cols;
	int n;
    for (n=0;n<size;++n){
    	copyTo[n] =orig[n];
    }
	
    return copyTo;
}

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 2 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*      !!  SPECIAL NOTE ON GROUP 2 FUNCTIONS: These functions have been written in such a way    */
/*          that the manipulated data from the input matrix can also be passed out to the same    */
/*          matrix.                                                                               */
/*==================================================================================================
 *  fix
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      implicitly returns a matrix through variable "outmatrix."
 *
 *  Description: truncates all elements in "matrix" to integer. So, this function rounds all matrix
 *  elements toward 0.
 *
 */
void fix(data_t *outmatrix, data_t *matrix, int rows, int cols) {
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)  outmatrix[i * cols + j] = (data_t)((int)matrix[i * cols + j]);
    }
    
    return;
}

/*==================================================================================================
 *  fliplr
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = Rows
 *      value, type integer         = Cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a pointer to a matrix through variable "outmatrix."
 *
 *  Description: This function takes the inputs, Rows and Cols, and returns a data_t pointer
 *  to a matrix that holds the orignal matrix's columns fliped from left to right . The dimensions
 *  of the matrix will be Rows X Cols.
 *          [1 2]   =  [2 1]
 *          [3 4]      [4 3]
 *  
 */
void fliplr(data_t *outmatrix, data_t *matrix, int rows, int cols){
	int i, j;
	int t = cols - 1;
    data_t temp;
	
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols / 2; j++) {
            temp = matrix[i * cols + t];
            matrix[i * cols + t] = outmatrix[i * cols + j];
            outmatrix[i * cols + j] = temp;
			t--;
		}
		t = cols - 1;	
    }
    
    return;
}
 
/*==================================================================================================
 *  reorder_matrix
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *      double pointer, type data_t = rowVect
 *
 *  Returns
 *      N/A
 *      implicitly returns a matrix through variable "outmatrix."
 *
 *  Description: This function reorders the columns of matrix based on the values of elements in 
 *  "rowVect." The length of the vector must = "cols," and all elements of rowVect must be unique
 *  and fall in the range [0, cols - 1].
 *      [ 1  2  3  4  5  6]                     [ 5  3  1  4  2  6]
 *      [ 7  8  9 10 11 12] ~ [4 2 0 3 1 5] =>  [11  9  7 10  8 12]
 *      [13 14 15 16 17 18]                     [17 15 13 16 14 18]
 *
 */
void reorder_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t *rowVect) {
    
    int i, j;
    data_t *dummy;
    
    allocate_matrix(&dummy, rows, cols);
    
    for (j = 0; j < cols; j++) {
        for (i = 0; i < rows; i++)  dummy[i * cols + j] = matrix[i * cols + (int)rowVect[j]];
    }
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)  matrix [i * cols + j] = dummy[i * cols + j];
    }
    
    free_matrix(&dummy);
    
    return;
}

/*==================================================================================================
 *  matrix_acos
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = Rows
 *      value, type integer         = Cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a pointer to a matrix through variable "outmatrix."
 *
 *  Description: This function takes the inputs, Rows and Cols, and returns a double pointer
 *  to a matrix that holds the orignal matrix with acos applied (all numbers in matrix must be <= 1.
 *  The dimensions of the matrix will be Rows X Cols.
 *
 */
 void matrix_acos(data_t *outmatrix, data_t *matrix, int rows, int cols){
	int i, j;
	
	for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = acos(matrix[i * cols + j]);
		}
	}
	
	return;
 }

/*==================================================================================================
 *  raise_matrix_to_power
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *      value, type integer         = power
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'powmatrix.'
 *
 *  Description: This function raises every element in an array to the degree passed in variable
 *  'power.'
 *      [X X X X]       [X^y X^y X^y X^y]
 *      [X X X X]   =>  [X^y X^y X^y X^y]
 *      [X X X X]       [X^y X^y X^y X^y]
 *
 */
void raise_matrix_to_power(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar) {
	int i,j;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = pow(matrix[i * cols + j], scalar);
		}
	}
	
	return;
}

/*==================================================================================================
 *  scale_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = scalar
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a vector through variable 'outmatrix.'
 *
 *  Description: Scales variable 'matrix' by variable 'scalar' and stores result in 'outmatrix.'
 *               [x x x]                             [xy xy xy]
 *      matrix = [x x x]    scalar = y  =>  matrix = [xy xy xy]
 *               [x x x]                             [xy xy xy]
 *
 */
void scale_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int scalar) {
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)  outmatrix[i * cols + j] = matrix[i * cols + j] * scalar;
    }
    
    return;
}

/*==================================================================================================
 *  divide_by_constant
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type data_t          = divisor
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'quotmatrix.'
 *
 *  Description: This function divides every element of the initial matrix by the value in the
 *  variable 'divisor.'
 *      [X X X X]       [X/y X/y X/y X/y]
 *      [X X X X]   =>  [X/y X/y X/y X/y]
 *      [X X X X]       [X/y X/y X/y X/y]
 *
 */
void divide_by_constant(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar) {
	int i, j;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = matrix[i * cols + j] / scalar;
		}
	}
	
	return;
}
/*==================================================================================================
 *  divide_scaler_by_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type data_t          = divisor
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'quotmatrix.'
 *
 *  Description: This function divides every element of the initial matrix by the value in the
 *  variable 'divisor.'
 *      [X X X X]       [y/X y/X y/X y/X]
 *      [X X X X]   =>  [y/X y/X y/X y/X]
 *      [X X X X]       [y/X y/X y/X y/X]
 *
 */
void divide_scaler_by_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar) {
	int i, j;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = scalar / matrix[i * cols + j];
		}
	}
	
	return;
}

/*==================================================================================================
 *  sum_scalar_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type data_t          = scalar
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: Adds a scalar value to all elements in the initial matrix
 *      [X X X X]       [X+y X+y X+y X+y]
 *      [X X X X]   =>  [X+y X+y X+y X+y]
 *      [X X X X]       [X+y X+y X+y X+y]
 *
 */
void sum_scalar_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols, data_t scalar) {
	int i,j;
	
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = matrix[i * cols + j] + scalar;
		}
	}
	
	return;
}

/*==================================================================================================
 *  normalize
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: 
 *
 */
void normalize(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i, j;
	data_t max, min;
		
	/*  Find max and min    */
	max = matrix[0];
	min = matrix[0];
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			if(matrix[i * cols + j] < min) min = matrix[i * cols + j];
			
			if(matrix[i * cols + j] > max) max = matrix[i * cols + j];
		}
	}
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			outmatrix[i * cols + j] = ((matrix[i * cols + j] - min) / (max-min));
		}
	}
	
	return;
}

/*==================================================================================================
 *  matrix_sqrt
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix'
 *
 *  Description: 
 *
 */
void matrix_sqrt(data_t *outmatrix, data_t *matrix, int rows, int cols){
int i, j;

	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			outmatrix[i * cols + j] = sqrt(matrix[i * cols + j]);
		}
	}
}

/*==================================================================================================
 *  matrix_sqrtm
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix'
 *
 *  Description: this finds the eig values and vectors square roots them and 
 *               then multiplies (eig vectors * eig value) / (eig vectors)
 *
 */

void matrix_sqrtm(data_t *outmatrix, data_t *matrix, int rows, int cols){
	data_t *eig_vect;
	data_t *eig_vals;
	data_t *temp;
	data_t *temp1;
	data_t *temp2;
	data_t *s;
	
	allocate_matrix(&eig_vect, rows, cols);
	allocate_vector(&eig_vals, cols);
	allocate_matrix(&temp, rows, cols);
	allocate_matrix(&temp1, rows, rows);
	allocate_matrix(&temp2, rows, cols);
	allocate_matrix(&s, rows, cols);


    matrix_eig(eig_vect, eig_vals, matrix, rows, cols); 
	//eig_vals? in next call?	
	matrix_sqrt(temp, eig_vals, 1, cols);
	multiply_matrices(temp1, eig_vect,  temp, rows, cols, rows);
	matrix_division(outmatrix, temp1, eig_vect, rows, cols, rows, cols);
	
	//free_matrix(&temp);
	//free_matrix(&temp1);
	//free_matrix(&temp2);
	//free_matrix(&eig_vect);
	//free_vector(&eig_vals);
	
	return;

}

/*==================================================================================================
 *  matrix_eig
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns eig vectors though a matrix through variable 'out_eig_vect'
 * 		Implicitly returns eig vals through a matrix through varaible 'out_eig_vals
 *		
 *  Description: this finds the eig values and vectors using the lapack
 *
 */
 
void matrix_eig(data_t *out_eig_vect, data_t *out_eig_vals, data_t *matrix, int rows, int cols){
	//int info;
	//data_t *workmatrix *tempMat;
	  data_t *dummy, *dummy2 = NULL;
	
	//tempMat = copy(matrix, rows, cols);
	//allocate_matrix(&workmatrix, rows, cols);
	
    allocate_vector(&dummy, cols);
    allocate_matrix(&dummy2, rows, cols);
    
	//original
	//DGEEV('N', 'V', cols, matrix, rows, out_eig_vals, dummy, dummy2, 0, out_eig_vect, cols + 1, workmatrix, -1, info);
   
	//LAPACKE_dgeev(rows * cols, 'N', 'V', cols, matrix, rows, out_eig_vals, dummy, dummy2, 0, out_eig_vect, cols + 1);
    // N means NO and V means Yes
	//*********LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', cols, matrix, rows, out_eig_vals, dummy, dummy2, rows, out_eig_vect, cols + 1);
    
	/*testing take out later*/
	printf("\n out_eig_vals\n");
	print_matrix(out_eig_vals, rows, cols);
	printf("\n dummy\n");
	print_matrix(dummy, rows, cols);
	printf("\n dummy2\n");
	print_matrix(dummy2, rows, cols);
	printf("\n out_eig_vect\n");
	print_matrix(out_eig_vect, rows, cols);
	printf("\nend eig testing \n\n");
    //free_matrix(&tempMat);
   // free_matrix(&workmatrix);
   // free_matrix(&dummy2);
    //free_vector(&dummy);
}

 
 /*==================================================================================================
 *  matrix_negate
 
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through outmatrix
 * 			
 *  Description: Negates a matrix
 *
 */
 void matrix_negate(data_t *outmatrix, data_t *matrix, int rows, int cols){
 	int i, j;	
		for(i = 0; i < rows; i++){
			for(j = 0; j < cols; j++){
				outmatrix[i * cols + j] = -(matrix[i * cols + j]);
			}
		}	
 }
 
 
 /*==================================================================================================
 *  matrix_exp
 
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through outmatrix
 * 			
 *  Description: takes component wise exponential of each function
 *      y = e^(i*pi)
 */
 void matrix_exp(data_t *outmatrix, data_t *matrix, int rows, int cols){
	 int i, j;	
		
		for(i = 0; i < rows; i++){
			for(j = 0; j < cols; j++){
				outmatrix[i * cols + j] = exp((matrix[i * cols + j])); //* M_PI);
			}
		}
 
 
}
/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 3 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */

/*==================================================================================================
 *  transpose
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'tmatrix.'
 *
 *  Description: This function takes a matrix and returns its transpose.
 *      tmatrix = matrix'
 *  The initial matrix has dimensions rows X cols so the resulting matrix has dimensions
 *  cols X rows.
 *
 */
void transpose(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i,j;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++)   outmatrix[j * rows + i] = matrix[i * cols + j];
	}

	return;
}

/*==================================================================================================
 *  mean_of_matrix
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'meanmatrix.'
 *
 *  Description: Builds a vector of the average of each column of a matrix. The dimension of the 
 *  initial matrix is rows X cols and the resulting vector has dimensions 1 X cols.
 *      [X X X X]
 *      [X X X X]   =>  [3X 3X 3X 3X] * 1/3 =>  [X X X X]
 *      [X X X X]
 *
 */
void mean_of_matrix(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i;
		
	sum_columns(outmatrix, matrix, rows, cols);
	
	for (i = 0; i < cols; i++)  outmatrix[i] /= (float)rows;
		
	return;
}

/*==================================================================================================
 *  reshape
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type data_t          = scalar
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: Adds a scalar value to all elements in the initial matrix
 *      [X X X X]       [X+y X+y X+y X+y]
 *      [X X X X]   =>  [X+y X+y X+y X+y]
 *      [X X X X]       [X+y X+y X+y X+y]
 *
 */
void reshape(data_t **outmatrix, int outRows, int outCols, data_t **matrix, int rows, int cols) {
    int i, j;
    
    if (outRows * outCols != rows * cols) {
        printf("Can not reorder the matrix\n");
        return;
    }
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            outmatrix[(i * cols + j) / outCols][(i * cols + j) % outCols] = matrix[i][j];
        }
    }
    
    return;
}

/*==================================================================================================
 *  mean_of_matrix_by_rows
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'meanmatrix.'
 *
 *  Description: Builds a vector of the average of each row of a matrix. The dimension of the 
 *  initial matrix is rows X cols and the resulting vector has dimensions rows X 1.
 *      [X X X X]		[4X]			[X]
 *      [X X X X]   =>  [4X] * 1/4 =>   [X]
 *      [X X X X]		[4X]			[X]
 *
 */
void mean_of_matrix_by_rows(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i;
		
	sum_rows(outmatrix, matrix, rows, cols);
	
	for (i = 0; i < rows; i++)  outmatrix[i] /= (float)cols;
		
	return;
}

/*==================================================================================================
 *  find
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a vector through variable 'outvect.'
 *
 *  Description: Finds all non-zero elements in a vector and returns the indices in another vector.
 *
 */
void find(data_t *outvect, data_t **matrix, int rows, int cols) {
	int i,j, count = 0;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			if(matrix[i][j] != 0) {
				outvect[count] = i;
				count++;
			}
		}
	}
	
	return;
}

/*==================================================================================================
 *  sum_rows
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function sums all the elements in every row and stores them in a column
 *  vector. So the input matrix has dimensions rows X cols and the resulting vector has dimension
 *  rows X 1.
 *      [X X X X]       [4X]
 *      [X X X X]   =>  [4X]
 *      [X X X X]       [4X]
 *
 */
void sum_rows(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i, j;
	
	for (i = 0; i < rows; i++) {
		outmatrix[i] = 0;
		
		for (j = 0; j < cols; j++)  outmatrix[i] += matrix[i * cols + j];
	}
	
	return;
}

/*==================================================================================================
 *  sum_columns
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function sums all the elements in every column and stores them in a row
 *  vector. So the input matrix has dimensions rows X cols and the resulting vector has dimension
 *  1 X cols.
 *      [X X X X]
 *      [X X X X]   =>  [3X 3X 3X 3X]
 *      [X X X X]
 *
 */
void sum_columns(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i, j; 

	for (j = 0; j < cols; j++) {
	    outmatrix[j] = 0;
	    
		for (i = 0; i < rows; i++)  outmatrix[j] += matrix[i * cols + j];
	}
	
	return;
}

/*==================================================================================================
 *  norm
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      data_t pointer, type data_t
 *
 *  Description: Returns the normalized length, which is the square root of the sum of the squares
 *  of the elements in the matrix.
 *
 */
data_t norm(data_t *matrix, int rows, int cols) {
	int i, j;
	data_t sum = 0, result = 0;
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			sum += (matrix[i * cols + j] * matrix[i * cols + j]);   /*  square the value    */
		}
	}
	
	result = sqrt(sum);
	
	return result;  
}

/*==================================================================================================
 *  determinant
 *
 *  Parameters
 *      single pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a vector through variable "vector."
 *
 *  Description: Returns determinant of matrix.
 */
void determinant(data_t *matrix, int rows, double *determ) {
    int i, j, j1, j2;
    double det = 0;
    double *m = NULL;

    if (rows < 1)   printf("error finding determinant\n");
    else if (rows == 1) det = matrix[0]; /* Shouldn't get used */
    else if (rows == 2) det = matrix[0] * matrix[1 * rows + 1] - matrix[1 * rows] * matrix[1];
    else {
        det = 0;
        for (j1 = 0; j1 < rows; j1++) {
            m = malloc((rows-1) * (rows - 1) * sizeof(double *));
            for (i = 1; i < rows; i++) {
                j2 = 0;
                for (j = 0; j < rows; j++) {
                    if (j == j1)    continue;
                    m[(i - 1) * (rows - 1) + j2] = matrix[i * rows + j];
                    j2++;
                }
            }
            determinant(m, rows - 1, determ);
            det += pow(-1.0, j1 + 2.0) * matrix[0 * rows + j1] * (*determ);
            free(m);
        }
    }
    *determ = (data_t)det;
    
    return;
}

/*==================================================================================================
 *  inv
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix'
 *
 *  Description: 
 *		modified parts of code form http://www.cs.rochester.edu/~brown/Crypto/assts/projects/adj.html
 */
void inv(data_t *outmatrix, data_t *matrix, int rows) {
	double det = 0;
	
	data_t *temp;
	
	allocate_matrix(&temp, rows, rows);
	
	cofactor(outmatrix, matrix, rows);
	determinant(matrix, rows, &det);

	divide_by_constant(outmatrix, outmatrix, rows, rows, det);
	
	free_matrix(&temp);
	
	return;
}

/*==================================================================================================
 *  Cofactor
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix'
 *
 *  Description: 
 *		finds cofactor of matrix
 *		modified parts of code form http://www.cs.rochester.edu/~brown/Crypto/assts/projects/adj.html
 */
void cofactor(data_t *outmatrix, data_t *matrix, int rows) {
    int i, j, ii, jj, i1, j1;
    double det;
    double *c;

    c = malloc((rows - 1) * (rows - 1) * sizeof(double *));
   
    for (j = 0; j < rows; j++) {
        for (i = 0; i < rows; i++) {
            /* Form the adjoint a_ij */
            i1 = 0;
            for (ii = 0; ii < rows; ii++) {
                if (ii == i)    continue;
                j1 = 0;
                for (jj = 0; jj < rows; jj++) {
                    if (jj == j) continue;
                    c[i1 * (rows - 1) + j1] = matrix[ii * rows + jj];
                    j1++;
                }
                i1++;
            }

            /* Calculate the determinant */
            determinant(c, rows-1, &det);

            /* Fill in the elements of the cofactor */
            outmatrix[j * rows + i] = pow(-1.0,i+j+2.0) * det;
        }
    }
   
   free(c);
   return;
}

/*==================================================================================================
 *  covariance
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: 
 *
 */
void covariance(data_t *outmatrix, data_t *matrix, int rows, int cols) {
	int i, j, k;
	
	data_t *average, *norm;
	data_t temp;
	allocate_matrix(&average, 1, cols);
	allocate_matrix(&norm, rows, cols);
	
    mean_of_matrix(average, matrix, rows, cols);
    
    for (j = 0; j < cols; j++) {
        for (i = 0; i < rows; i++)
            norm[i * cols + j] = matrix[i * cols + j] - average[j];
    }
    
    for (j = 0; j < cols; j++) {
        for (k = 0; k < cols; k++) {
            temp = 0;
            for (i = 0; i < rows; i++)
                temp += norm[i * cols + j] * norm[i * cols + k];
                
            outmatrix[j * cols + k] = temp / (cols - 1);
        }
    }

    free_matrix(&average);
	free_matrix(&norm);
	return;
}

/*==================================================================================================
 *  submatrix
 *
 *  Parameters
 *      double pointer, type data_t = outmatrix
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *      value, type integer         = start_row
 *      value, type integer         = start_col
 *		value, type integer			= end_row
 *      value, type integer         = end_col
 *
 *
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: 
 *      This function returns a submatrix of the input matrix specified by an upper left corner
 *      (start_row, start_col) and a lower right corner (end_row, end_col). This is assuming
 *		matrix starts at row = 0 and col = 0.
 */
void submatrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int start_row, int start_col, int end_row, int end_col) {
    int i,j;
    int sub_rows, sub_cols;

    sub_rows = end_row - start_row + 1;
    sub_cols = end_col - start_col + 1;

    for (i = 0; i < sub_rows; i++) {
        for (j = 0; j < sub_cols; j++) {
            outmatrix[i * sub_cols + j] = matrix[(start_row + i) * cols + (start_col + j)];
        } 
    }

    return;
}

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 4 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*      !!  SPECIAL NOTE ON GROUP 4 FUNCTIONS: These functions have been written in such a way    */
/*          that the manipulated data from the input matrix can also be passed out to the same    */
/*          matrix.                                                                               */

/*==================================================================================================
 *  subtract_matrices
 *
 *  Parameters
 *      double pointer, type data_t = matrix1
 *      double pointer, type data_t = matrix2
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function subtracts the elements of matrix2 from the elements of matrix 1. So,
 *  matrix1[i][j] = matrix1[i][j] - matrix2[i][j]. 
 *      [X X X]     [Y Y Y]     [X-Y X-Y X-Y]
 *      [X X X] -   [Y Y Y] =>  [X-Y X-Y X-Y]
 *      [X X X]     [Y Y Y]     [X-Y X-Y X-Y]
 *
 */
void subtract_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols) {
    int i, j;
	
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            outmatrix[i * cols + j] = matrix1[i * cols + j] - matrix2[i * cols + j];
        }
    }
	
    return;
}
/*  add_matrices
 *
 *  Parameters
 *      double pointer, type data_t = matrix1
 *      double pointer, type data_t = matrix2
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function adds the elements of matrix2 from the elements of matrix 1. So,
 *  matrix1[i][j] = matrix1[i][j] + matrix2[i][j]. 
 *      [X X X]     [Y Y Y]     [X+Y X+Y X+Y]
 *      [X X X] +   [Y Y Y] =>  [X+Y X+Y X+Y]
 *      [X X X]     [Y Y Y]     [X+Y X+Y X+Y]
 *
 */
void add_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols) {
    int i, j;
	
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            outmatrix[i * cols + j] = matrix1[i * cols + j] + matrix2[i * cols + j];
        }
    }
	
    return;
}
 /*  matrix_dot_division
 *
 *  Parameters
 *      double pointer, type data_t = matrix1
 *      double pointer, type data_t = matrix2
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function divides the elements of matrix1 by the elements of matrix 2. So,
 *  matrix1[i][j] = matrix1[i][j] / matrix2[i][j]. 
 *      [X X X]     [Y Y Y]     [X/Y X/Y X/Y]
 *      [X X X] /   [Y Y Y] =>  [X/Y X/Y X/Y]
 *      [X X X]     [Y Y Y]     [X/Y X/Y X/Y]
 *
 */
void matrix_dot_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols){
    int i, j;

	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			outmatrix[i * cols + j] = matrix1[i * cols + j] / matrix2[i * cols + j];
		}
	}

    return;
}

 /*  matrix_division
 *
 *  Parameters
 *      double pointer, type data_t = matrix1
 *      double pointer, type data_t = matrix2
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: This function multiples the elements of matrix1 times the inv of matrix 2. So,
 *  matrix1 = matrix1 * (matrix2)^-1. 
 *      [X X X]     [Y Y Y] -1   [X*Y^-1 X*Y^-1 X*Y^-1]
 *      [X X X]  *  [Y Y Y]  =>  [X*Y^-1 X*Y^-1 X*Y^-1]
 *      [X X X]     [Y Y Y]      [X*Y^-1 X*Y^-1 X*Y^-1]
 *
 */
void matrix_division(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows1, int cols1, int rows2, int cols2){
	data_t *temp;
	allocate_matrix(&temp, rows2, cols2);
	
	inv(temp, matrix2, rows2);
	multiply_matrices(outmatrix, matrix1, temp, rows1, cols2, rows2);
	
	free_matrix(&temp);
    
	return;
}

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GROUP 5 FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  */

/*==================================================================================================
 *  multiply_matrices
 *
 *  Parameters
 *      double pointer, type data_t = matrix1
 *      double pointer, type data_t = matrix2
 *      value, type integer         = rows
 *      value, type integer         = cols
 *      value, type integer         = k
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'outmatrix.'
 *
 *  Description: Multiplies two matrices together. The first matrix has dimensions rows X k. The
 *  second has dimensions k X cols. The resulting matrix has dimensions rows X cols. Note that
 *  cols1 and rows2 MUST be equivalent.
 *      [X X X]     [Y Y]
 *      [X X X] *   [Y Y]   =>  [3XY 3XY]
 *                  [Y Y]       [3XY 3XY]
 *
 */
void multiply_matrices(data_t *outmatrix, data_t *matrix1, data_t *matrix2, int rows, int cols, 
    int k) {
	
	int a,b,c;
	data_t sum;
	
	/*	Multiply matrices	*/
	for (a = 0; a < rows; a++) {
		for (b = 0; b < cols; b++) {
			sum = 0;
			
			for (c = 0; c < k; c++) sum += matrix1[a * k + c] * matrix2[c * k + b];
		    outmatrix[a * rows + b] = sum;
		}
	}
	
	return;
}

#endif
