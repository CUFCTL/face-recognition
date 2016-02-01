/*==================================================================================================
 *  pcabigFn.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *  
 *  This file contains
 *      pcabigFn
 *  
 *  Lasted Edited: Jul. 3, 2013
 *
 *  Changes made: major changes to last section 
 *
 */

 #include "levelTwoOps.c"
 #include "matrix_manipulation.h" 
/*==================================================================================================
 *  pcabigFn
 *
 *  Parameters
 *      double pointer, type double = matrix
 *      value, type integer         = rows
 *      value, type integer         = cols
 *
 *  Returns
 *      N/A
 *      Implicitly returns a matrix through variable 'matrix.'
 *
 *  Description: WHAT NEEDS WORK
 *      LINE 94 - figure out matrix division
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      
 */
void pcabigFn(data_t **U, data_t **R, data_t **E, int rows, int cols, data_t **B, int num_pixels, int num_images) {
                                                //  %function [U,R,E] = pcabigFn(B);
                                                //  %Compute PCA by calculating smaller covariance matrix and reconstructing
                                                //  %eigenvectors of large cov matrix by linear combinations of original data
                                                //  %given by the eigenvecs of the smaller cov matrix. 
                                                //  %Data in Cols of B. Third version.  
                                                //  %
                                                //  %***** justification
                                                //  %
                                                //  %B = N x P data matrix.  N = dim of data  (Data in Cols of B, zero mean)
                                                //  %                        P = #examples
                                                //  %                        N >> P
                                                //  %
                                                //  %Want eigenvectors ui of BB' (NxN)
                                                //  %Solution:
                                                //  %Find eigenvectors vi of B'B (PxP)
                                                //  %From def of eigenvector: B'Bvi = di vi ---> BB'Bvi = di Bvi
                                                //  %Eigenvecs of BB' are Bvi
                                                //  %-------------------------------
                                                //  %[V,D] = eig (B'B)
                                                //  %Eigenvecs are in cols of V.    (Sorted cols)
                                                //  %
                                                //  %U = BV;  Cols of U are Bvi (eigenvecs of lg cov mat.) (Gave unit length)
                                                //  %R = B'U; Rows of R are pcarep of each observation.
                                                //  %E = eigenvalues        (eigenvals of small and large cov mats are equal)
                                                //  %*****
                                                //
                                                //  function [U,R,E] = pcabigFn(B)
                                                //
												//  %Read data into columns of B;
    data_t *length_ones;
	data_t *B_zeromean;
	data_t *B_zmtrans;
	data_t *B_mult;						//  %B = datamat';
	data_t *B_div;
	data_t *temp;
	data_t *temp_vec;
	data_t *index;
	data_t *Vsort;
	data_t *U_squared;
	
	data_t *righteignm;
	data_t *length_matrix;
	//data_t *wr_matrix;
	data_t *wr;
	data_t *wi;
	data_t *lefteigenvect;
	data_t *righteigenvect;
	data_t *workmatrix;
	data_t *B_vector;
	int integer;
	
	/*  Allocation of all arrays    */
	allocate_matrix(&B_zeromean, num_pixels, num_images);
	// Rows by 1 array of ones
	allocate_matrix(&length_ones, rows, 1);
	// temp vars are used to move matrix from on function to another
	allocate_matrix(&temp_vec, 1, cols);
	
    allocate_matrix(&index, 1, cols);			//  [N,P] = size(B);    -   sizes found in matlab code                 
	allocate_matrix(&Vsort, num_images, num_images);
	// image data transposed
	allocate_matrix(&B_zmtrans, num_images, num_pixels);
	allocate_matrix(&B_mult, num_images, num_images);
	allocate_matrix(&B_div, num_images, num_images);
	allocate_vector(&B_vector, num_images*num_images);
	
	allocate_matrix(&U, num_pixels, num_images);
	allocate_matrix(&U_squared, num_pixels, num_images);
	allocate_matrix(&righteignm, cols, cols);
	allocate_matrix(&length_matrix, rows, cols);
	allocate_matrix(&R, cols, cols);
	//allocate_matrix(&wr_matrix, cols, cols);
	
	
	// there seems to be a problem here
	allocate_matrix(&wr, cols);
	allocate_matrix(&wi, cols);
	allocate_vector(&lefteigenvect, num_images);
	allocate_vector(&righteigenvect, num_images);
	allocate_matrix(&workmatrix, cols);

	/*  END OF ARRAY ALLOCATION */
	
    /*  zero_mean() subtracts out the mean  */  //  %********subtract mean
    zero_mean(B_zeromean, B, num_pixels, num_images);      //  mb=mean(B');
                                                //  B=B-(ones(P,1)*mb)';
                                                //
												//  %********Find eigenvectors vi of B'B (PxP)
                                                //  [V,D] = eig (1/(P-1)*(B'*B));   %scale factor gives eigvals correct
    transpose(B_zmtrans, B_zeromean, num_pixels, num_images);
	multiply_matrices(B_mult, B_zmtrans, B_zeromean, num_images, num_images, num_pixels);
	divide_by_constant(B_div, B_mult, num_images, num_images, (data_t)num_images - 1);
	matrix_to_vector(B_vector, B_div, num_images, num_images);
	
	matrix_eig(righteignm, wr?, B_vector, num_pixels, num_images);
	//DGEEV('N', 'V', cols, vector, cols, wr, wi, lefteigenvect, 0, righteigenvect, cols + 1, workmatrix, -1, integer);
    /*  from lapacke    */                      //  %magnitude for large cov mat 
												//  %(assuming sample cov)
                                                //  %********Sort eigenvectors
                                                //  eigvalvec = max(D); -   handled by lapack function
	                                        	//  [seigvals, index] = sort(eigvalvec); % sort goes low to high
	eigSort(righteignm, (data_t*)wr, num_images, num_pixels);
	//fliplr(index, index, rows, cols);
	
	fliplr(Vsort, righteignm, num_images, num_images);
												//Vsort = V(:,fliplr(index));
    /*  !will use mergesort to sort eigenvalues... when two eigenvalues are switched    !
        !the corresponding eigenvectors also will need to be switched                   !   */

                                                //  %********Reconstruct
    multiply_matrices(U, B_zeromean, Vsort, num_pixels, num_images, num_images);        //  U = B*Vsort;  % Cols of U are Bvi. (N-dim Eigvecs)
                                                //
                                                //  %********Give eigvecs unit length.  Improves classification.
												//  length = sqrt (sum (U.^2)); 
												//temp = sum_along_rows(temp); <<worng?
	raise_matrix_to_power(U_squared, U, num_pixels, num_images, 2);
	sum_columns(temp, temp_vec, rows, cols);
	matrix_sqrt(temp_vec, temp_vec, rows, cols);
    
	ones(length_ones, rows, 1);					//  U = U ./ (ones(N,1) * length);
	multiply_matrices(length_matrix, length_ones, temp_vec, rows, cols, 1);
	matrix_dot_division(U, U, length_ones, rows, cols); 
	multiply_matrices(R, matrix_t, U, cols, cols, rows);      //  R = B'*U;  % Rows of R are pcarep of each image.
    
	// IS wr matrix = sorted eigvals but not set convert wr to matrix and here?
	//fliplr(wr, wr, cols, cols);          //  E = fliplr(seigvals); 
	fliplr(E, wr, cols, cols);
	
	free_matrix(&temp);
	free_matrix(&temp_vec);
	free_matrix(&workmatrix);
    return;
}