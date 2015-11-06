/*==================================================================================================
 *  levelTwoOps.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *
 *  This file contains
 *		zero_mean
 *      cosFn
 *      eigSort
 *
 *  Lasted Edited: Jul. 10, 2013
 *
 *  Changes made: by William - changed several function calls to reflect changes in 
 *  "matrix_manipulation.h" prototypes.
 *
 */
 
/*	For all of these functions, remember that you cannot actually pass a 3D matrix into a function
	Because the FPGAs will be using sequential code instead of functions, this should not be a problem.
	The arguements simply show what you will need to complete each function.
 */

#include "matrix_manipulation.h"

/*==================================================================================================
 *  zero_mean
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      double pointer, type data_t
 *
 *  Description: Returns a zero-mean form of the matrix X. Each row of Xzm will have 
 *	zero mean, same as in spherex.m. For PCA, put the observations in cols
 *	before doing zeroMn(X).
 *
 *	function Xzm = zeroMn(X)
 *	
 *	[N,P] = size(X);
 *	
 *	mx=mean(X');
 *	
 *	Xzm=X-(ones(P,1)*mx)';
 *
 *  THIS FUNCTION CALLS
 *      transpose           (matrix_manipulation.c)
 *      mean_of_matrix      (matrix_manipulation.c)
 *      multiply_matrices   (matrix_manipulation.c)
 *      subtract_matrices   (matrix_manipulation.c)
 *      
 *  THIS FUNCTION IS CALLED BY
 *      zero_mean   (zeroMn.c)
 *
 */
void zero_mean(data_t *zeromean, data_t *matrix, int rows, int cols) {
	
	data_t *tmatrix;       /*  [cols]  [rows]    */
    data_t *mones;         /*  [cols]  [1]       */
    data_t *meanmatrix;    /*  [1]     [rows]    */
    data_t *submat;        /*  [rows]  [cols]    */
    
    allocate_matrix(&tmatrix, cols, rows);
    allocate_matrix(&mones, cols, 1);
    allocate_matrix(&meanmatrix, 1, rows);
    allocate_matrix(&submat, rows, cols);
    
    transpose(tmatrix, matrix, rows, cols);                             /*  [rows][cols]    =>  [cols][rows]            */
	mean_of_matrix(meanmatrix, tmatrix, cols, rows);                    /*  [cols][rows]    =>  [1][rows]               */

	/*  Populate the mones array with ones  */
	ones(mones, cols, 1);

	multiply_matrices(tmatrix, mones, meanmatrix, cols, rows, 1);       /*  [cols][1] * [1][rows]   =>  [cols][rows]    */
	transpose(submat, tmatrix, cols, rows);                             /*  [cols][rows]    =>  [rows][cols]            */

    subtract_matrices(zeromean, matrix, submat, rows, cols);            /*  [rows][cols] - [rows][cols] =>  [rows][cols]*/
	
	return;
}

/*==================================================================================================
 *  cosFn
 *
 *  Parameters
 *      double pointer, type data_t = mat1
 *      double pointer, type data_t = mat2
 *      value, type integer         = mat1_r
 *      value, type integer         = mat1_c
 *      value, type integer         = mat2_r
 *      value, type integer         = mat2_c
 *
 *  Returns
 *      double pointer, type data_t
 *
 *  Description: function [S] = cosFn(mat1,mat2),
 *	Computes the cosine (normalized dot product) between training vectors in 
 *	columns of mat1 and test vectors in columns of mat2. Outputs a matrix of 
 *	cosines (similarity matrix). 
 *
 *	function [S] = cosFn(mat1,mat2)  
 *		denom = sum(mat1.^2,1)*sum(mat2'.^2,2)
 *		denom (if(denom == 0)) = 0.00000000000000000000001;
 *		
 *		numer = mat1*mat2';
 *		
 *		S = numer./denom;
 *
 *	B = sum(A,dim) sums along the dimension of A specified by scalar dim. 
 *	The dim input is an integer value from 1 to N, where N is the number of dimensions in A.
 *	Set dim to 1 to compute the sum of each column, 2 to sum rows, etc.
 *
 *  THIS FUNCTION CALLS
 *      transpose                   (matrix_manipulation.c)
 *      raise_matrix_to_power       (matrix_manipulation.c)
 *      sum_matrix_along_columns    (matrix_manipulation.c)
 *      sum_matrix_along_rows       (matrix_manipulation.c)
 *      multiply_matrices           (matrix_manipulation.c)
 *      divide_by_constant          (matrix_manipulation.c)
 *
 *  THIS FUNCTION IS CALLED BY
 *
 */
void cosFn(data_t *output, data_t *mat1, data_t *mat2, int mat1_r, int mat1_c, int mat2_r, 
    int mat2_c) {

	/*  All arrays must be dynamically allocated    */
	data_t *denominator;
	
	/*  trans2 is mat2_cXmat2_r,  numerator is  mat1_rXmat2_r */
	data_t *trans2, *numerator;
    
    /*  powmatrix1 is mat1_rXmat1_c, powmatrix2 is mat2_cXmat2_r    */
    data_t *powmatrix1, *powmatrix2;
    
    /*  SPECIAL NOTE: matsums1 and matsums2 are both one dimensional vectors and are, however, they
        are represented by a double pointer. matsums1 is a 1Xmat1_c vector and matsums2 is an
        mat2_cX1 matrix. */
    data_t *matsums1, *matsums2;

    /*  Allocation of all arrays    */
    allocate_matrix(&trans2, mat2_c, mat2_r);
    allocate_matrix(&numerator, mat1_r, mat2_r);
    allocate_matrix(&powmatrix1, mat1_r, mat1_c);
    allocate_matrix(&powmatrix2, mat2_c, mat2_r);
    allocate_matrix(&matsums1, 1, mat1_c);
    allocate_matrix(&matsums2, mat2_c, 1);
    allocate_matrix(&denominator, 1, 1);
    /*  END OF ARRAY ALLOCATION */
    
    /*  Create transpose and get norm values    */
	transpose(trans2, mat2, mat2_r, mat2_c);
	raise_matrix_to_power(powmatrix1, mat1, mat1_r, mat1_c, 2);
	raise_matrix_to_power(powmatrix2, trans2, mat2_r, mat2_c, 2);
    
	sum_columns(matsums1, powmatrix1, mat1_r, mat1_c);
	sum_rows(matsums2, powmatrix2, mat2_r, mat2_c);
    
    /*  build denominator and check output to prevent division by 0 */
	multiply_matrices(denominator, matsums1, matsums2, 1, 1, mat1_r);
    
    if(denominator[0] == 0) denominator[0] = 0.0000000000000000001;
  
    /*  build numerator */
    multiply_matrices(numerator, mat1, trans2, mat1_c, mat2_c, mat1_r);
  
    /*  build final output by numerator / denominator   */
    divide_by_constant(output, numerator, mat1_c, mat2_c, denominator[0]);

	return;
}

/*==================================================================================================
 *  eigSort
 *
 *  Parameters
 *      double pointer, type data_t = matrix
 *      double pointer, type data_t = vector
 *      value, type integer         = size
 *
 *  Returns
 *      N/A
 *
 *  Description: This function is a recursive implementation of the merge sort algorithm. When
 *  completed, "vector" will be sorted in ascending order. Also manipulates the columns of "matrix"
 *  corresponding to the moved elements of "vector." So each column of "matrix" will be placed in 
 *  the same location as its corresponding element in "vector."
 *
 *  THIS FUNCTION CALLS
 *      
 *  THIS FUNCTION IS CALLED BY
 *      zero_mean   (zeroMn.c)
 *
 */
/*  recursive function MergeSort    */
void eigSort(data_t *matrix, data_t *vector, int size, int rows) {
	int i, j, k;        /*  loop counter    */
	int left, right;    /*  holds the starting indecies of partitioned array    */
	int roverL, roverR; /*  roving indecies: L used for left partition, R used for right    */
	data_t *temp;       /*  temporary array */
	data_t *tempMat;
	data_t *newMat;
	data_t *leftMat;
	
	/*  left partition always starts at zeros; right partition
	    always starts at floor(size/2)  */
	left = 0; 
	right = (int) floor(size / 2.0);
	int newMatCols = (int) ceil(size / 2.0);
	
	allocate_matrix(&temp, 1, size);
	allocate_matrix(&tempMat, rows, size);
	allocate_matrix(&newMat, rows, newMatCols);
	allocate_matrix(&leftMat, rows, right);
	
	for (i = 0; i < rows; i++) {
	    for (j = 0; j < newMatCols; j++)  newMat[i * newMatCols + j] = matrix[i * size + j + right];
	    for (k = 0; k < right; k++) leftMat[i * right + k] = matrix[i * size + k];
	}
	
	if (size > 2) {
		eigSort(leftMat, vector, right, rows);
		eigSort(newMat, &vector[right], newMatCols, rows);
		
		for (i = 0; i < rows; i++) {
	        for (j = 0; j < newMatCols; j++)    matrix[i * size + j + right] = newMat[i * newMatCols + j];
	        for (k = 0; k < right; k++) matrix[i * size + k] = leftMat[i * right + k];
	    }
	        
		/*  indexers through each partition */
		roverL = 0;
		roverR = 0;

		for (cnt = 0; cnt < size; cnt++) {
			/*  implies no more entries in left partition   */
			if (roverL == (int) floor(size / 2.0)) {
			    temp[cnt] = vector[right + roverR];
			    for (i = 0; i < rows; i++)  tempMat[i * size + cnt] = matrix[i * size + right + roverR];
				
				roverR++;
			
			/*  implies no more entries in right partition  */
			} else if (roverR == (int) ceil(size / 2.0)) {
				temp[cnt] = vector[roverL];
				for (i = 0; i < rows; i++)  tempMat[i * size + cnt] = matrix[i * size + roverL];

				roverL++;
			} else {
				if (vector[roverL] <= vector[right + roverR]) {
					temp[cnt] = vector[roverL];
					for (i = 0; i < rows; i++)  tempMat[i * size + cnt] = matrix[i * size + roverL];

					roverL++;
				} else if (vector[roverL] > vector[right + roverR]) {
					temp[cnt] = vector[right + roverR];
					for (i = 0; i < rows; i++)  tempMat[i * size + cnt] = matrix[i * size + right + roverR];

					roverR++;
				}
			}
		}
		
		for (cnt = 0; cnt < size; cnt++) {
		    vector[cnt] = temp[cnt];
		    
		    for (i = 0; i < rows; i++)  matrix[i * size + cnt] = tempMat[i * size + cnt];
		}

	/*  Base Case: if only two elements, compare them and sort 
		in ascending order. Note, size = 1 is also a base case
		but no computation needs to take place for this case.   */
	} else if (size == 2) {
		if (vector[0] > vector[1]) {
			temp[0] = vector[0];
			vector[0] = vector[1];
			vector[1] = temp[0];
			
			for (i = 0; i < rows; i++) {
			    tempMat[i * size] = matrix[i * size];
			    matrix[i * size] = matrix[i * size + 1];
			    matrix[i * size + 1] = tempMat[i * size];
		    }
		}
	}
	
	free_matrix(&temp);
	free_matrix(&tempMat);
	free_matrix(&newMat);
	
	return;
}
