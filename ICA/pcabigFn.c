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
 //    data_t *length_ones;
	// data_t *B_zeromean;
	// data_t *B_zmtrans;
	// data_t *B_mult;						//  %B = datamat';
	// data_t *B_div;
	//data_t *temp;
	data_t *temp_vec;
	// data_t *index;
	// data_t *Vsort;
	// data_t *U_squared;
	
	//data_t *righteignm;
	// data_t *length_matrix;
	// //data_t *wr_matrix;
	// data_t *wr;
	// data_t *wi;
	data_t *lefteigenvect;
	data_t *righteigenvect;
	//data_t *workmatrix;
	data_t *B_vector;
	int integer;
	
	/*  Allocation of all arrays    */
	//allocate_matrix(&B_zeromean, num_pixels, num_images);
	matrix_t *B_zeromean = m_intialize(UNDEFINED, num_pixels, num_images);

	// Rows by 1 array of ones
	//allocate_matrix(&length_ones, rows, 1);
	matrix_t *length_ones = m_intialize(UNDEFINED, rows, 1);
	// temp vars are used to move matrix from on function to another
	//allocate_matrix(&temp_vec, 1, cols);
	matrix_t *temp_vec = m_intialize(UNDEFINED, 1, cols);
	
    //allocate_matrix(&index, 1, cols);			//  [N,P] = size(B);    -   sizes found in matlab code                 
	matrix_t *index = m_intialize(UNDEFINED, 1, cols);

	//allocate_matrix(&Vsort, num_images, num_images);
	matrix_t *Vsort = m_intialize(UNDEFINED, num_images, num_images);
	// image data transposed

	//allocate_matrix(&B_zmtrans, num_images, num_pixels);
	matrix_t *B_zmtrans = m_intialize(UNDEFINED, num_images, num_pixels);

	//allocate_matrix(&B_mult, num_images, num_images);
	matrix_t *B_mult = m_intialize(UNDEFINED, num_images, num_images);

	//allocate_matrix(&B_div, num_images, num_images);
	matrix_t *B_div = m_intialize(UNDEFINED, num_images, num_images);

	//Needs to be re-written possibly or a function for initializing
	//vectors needs to be created for the shared code
	allocate_vector(&B_vector, num_images*num_images);


	//allocate_matrix(&U, num_pixels, num_images);
	matrix_t *U = m_intialize(UNDEFINED, num_pixels, num_images);

	//allocate_matrix(&U_squared, num_pixels, num_images);
	matrix_t *U_squared = m_intialize(UNDEFINED, num_pixels, num_images);

	//allocate_matrix(&righteignm, cols, cols);
	matrix_t *righteignm = m_intialize(UNDEFINED, cols, cols);

	//allocate_matrix(&length_matrix, rows, cols);
	matrix_t *length_matrix = m_intialize(UNDEFINED, rows, cols);

	//allocate_matrix(&R, cols, cols);
	matrix_t *R = m_intialize(UNDEFINED, cols, cols);
	//allocate_matrix(&wr_matrix, cols, cols);
	
	
	// there seems to be a problem here, don't know the value for rows to be passed to the function
	//allocate_matrix(&wr, cols);
	matrix_t *wr = m_intialize(UNDEFINED, ???, cols);
	//allocate_matrix(&wi, cols);
	matrix_t *wi = m_intialize(UNDEFINED, ???, cols);

	allocate_vector(&lefteigenvect, num_images);
	allocate_vector(&righteigenvect, num_images);
	
	//allocate_matrix(&workmatrix, cols);
	matrix_t *workmatrix = m_intialize(UNDEFINED, ???, cols);

	/*  END OF ARRAY ALLOCATION */
	
    /*  zero_mean() subtracts out the mean  */  //  %********subtract mean
    zero_mean(B_zeromean, B, num_pixels, num_images);      //  mb=mean(B');
                                                //  B=B-(ones(P,1)*mb)';
                                                //
												//  %********Find eigenvectors vi of B'B (PxP)
                                                //  [V,D] = eig (1/(P-1)*(B'*B));   %scale factor gives eigvals correct
    //transpose(B_zmtrans, B_zeromean, num_pixels, num_images);
    matrix_t *B_zmtrans = m_transpose(B_zeromean);

	//multiply_matrices(B_mult, B_zmtrans, B_zeromean, num_images, num_images, num_pixels);
	B_mult = m_matrix_multiply(B_zmtrans, B_zeromean);
	//multiply_matrices(outmatrix, matrix1, temp, rows1, cols2, rows2);
	matrix_t * outmatrix = m_matrix_multiply(matrix1, temp);

	//divide_by_constant(B_div, B_mult, num_images, num_images, (data_t)num_images - 1);
	B_div = m_elem_divideByConst(B_mult, (data_t)num_images - 1);

	//matrix_to_vector function is not declared in matrix_manipulation.h
	matrix_to_vector(B_vector, B_div, num_images, num_images);
	

	matrix_eig(righteignm, wr?, B_vector, num_pixels, num_images);
	//DGEEV('N', 'V', cols, vector, cols, wr, wi, lefteigenvect, 0, righteigenvect, cols + 1, workmatrix, -1, integer);
    /*  from lapacke    */                      //  %magnitude for large cov mat 
												//  %(assuming sample cov)
                                                //  %********Sort eigenvectors
                                                //  eigvalvec = max(D); -   handled by lapack function
	                                        	//  [seigvals, index] = sort(eigvalvec); % sort goes low to high
	//eigSort isn't in matrix_manipulation.h
	eigSort(righteignm, (data_t*)wr, num_images, num_pixels);
	//fliplr(index, index, rows, cols);
	
	//fliplr(Vsort, righteignm, num_images, num_images);
	Vsort = m_flipCols(righteignm);
												//Vsort = V(:,fliplr(index));
    /*  !will use mergesort to sort eigenvalues... when two eigenvalues are switched    !
        !the corresponding eigenvectors also will need to be switched                   !   */

                                                //  %********Reconstruct
    //multiply_matrices(U, B_zeromean, Vsort, num_pixels, num_images, num_images); 
    U = m_matrix_multiply(B_zeromean, Vsort);       //  U = B*Vsort;  % Cols of U are Bvi. (N-dim Eigvecs)
                                                //
                                                //  %********Give eigvecs unit length.  Improves classification.
												//  length = sqrt (sum (U.^2)); 
												//temp = sum_along_rows(temp); <<worng?
	//raise_matrix_to_power(U_squared, U, num_pixels, num_images, 2);
	U_squared = m_elem_pow(U, 2);

	//sum_columns(temp, temp_vec, rows, cols);
	temp = m_sumCols(temp_vec);
	//matrix_sqrt(temp_vec, temp_vec, rows, cols);
	temp_vec = m_elem_sqrt(temp_vec);
    
	//ones(length_ones, rows, 1);	
	length_ones = m_intialize(ONES, rows, 1);				//  U = U ./ (ones(N,1) * length);
	//multiply_matrices(length_matrix, length_ones, temp_vec, rows, cols, 1);
	length_matrix = m_matrix_multiply(length_ones, temp_vec);

	//matrix_dot_division(U, U, length_ones, rows, cols);
	U = m_dot_division(U, length_ones);

	//multiply_matrices(R, matrix_t, U, cols, cols, rows);
	R = m_matrix_multiply(matrix_t, U);      //  R = B'*U;  % Rows of R are pcarep of each image.
    
	// IS wr matrix = sorted eigvals but not set convert wr to matrix and here?
	//fliplr(wr, wr, cols, cols);          //  E = fliplr(seigvals); 
	//fliplr(E, wr, cols, cols);
	E = m_flipCols(wr);
	
	//free_matrix(&temp);
	m_free(temp);
	//free_matrix(&temp_vec);
	m_free(temp_vec);
	//free_matrix(&workmatrix);
	m_free(workmatrix);
    return;
}