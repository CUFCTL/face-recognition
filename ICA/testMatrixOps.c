/*==================================================================================================
 *  testMatrixOps.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *  
 *  This file contains
 *      main
 *  
 *  Lasted Edited: Jul. 2, 2013
 *
 *  Changes made: by William - changed several function calls to reflect the changes in several
 *  "matrix_manipulation.h" prototypes.
 *
 
 **** comand to run file 
 
	gcc testMatrixOps.c -lm  -L /home/CI_Biometric/lapacke/lapack-3.5.0 -I /home/CI_Biometric/lapacke/lapack-3.5.0/lapacke/include /home/CI_Biometric/lapacke/lapack-3.5.0/lapacke.a -o test
MatrixOps.o

 
 */

#include "matrix_manipulation.h"
#include <string.h>
#define ROWS 3
#define COLS 3

/*==================================================================================================
 *  main
 *
 *  Parameters
 *
 *  Returns
 *
 *  Description: 
 *
 *  THIS FUNCTION CALLS
 *
 */
int main() {
    data_t *matrix, *matrix_out, *Tmatrix, *colVect, *rowVect, *rowsXrows, *colsXcols, *one;
    int i, j;
	char skip[4] = "no";
    data_t deter;
    
    /*  allocation of matrices  */
    printf("Allocating memory...\n\n");
    allocate_matrix(&matrix, ROWS, COLS);
	allocate_matrix(&matrix_out, ROWS, COLS);
    allocate_matrix(&Tmatrix, COLS, ROWS);
    allocate_matrix(&colVect, ROWS, 1);
    allocate_matrix(&rowVect, 1, COLS);
    allocate_matrix(&rowsXrows, ROWS, ROWS);
    allocate_matrix(&colsXcols, COLS, COLS);
    allocate_matrix(&one, 1, 1);
    /*  End allocation  */
    
    /*  fill matrix test */
    printf("\n\nFilling Matrix...\n\n");
    printf("matrix = \n");
    fill_matrix(matrix, ROWS, COLS);
    print_matrix(matrix, ROWS, COLS);
	getchar();
    system("clear");
	
	printf("skip functions that have been tested (y/n)?");
	scanf("%c", skip);
	if(strcmp(skip, "yes")  < 0){
		/*  transpose test  */
		printf("\n\nTransposing Matrix...\n\n");
		transpose(Tmatrix, matrix, ROWS, COLS);
		printf("Tmatrix = \n");
		print_matrix(Tmatrix, COLS, ROWS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  sum along rows test */
		printf("\n\nSumming matrix along rows...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		sum_rows(colVect, matrix, ROWS, COLS);
		printf("colVect = \n");
		print_matrix(colVect, ROWS, 1);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
			
		/*  sum along column test   */
		printf("\n\nSumming matrix along columns...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		sum_columns(rowVect, matrix, ROWS, COLS);
		printf("rowVect = \n");
		print_matrix(rowVect, 1, COLS);
		printf("\n");
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  raise matrix to power test  */
		printf("\n\nRaising matrix elements to powers...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		raise_matrix_to_power(matrix, matrix, ROWS, COLS, 2);
		printf("powmatrix = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  mean of matrix test */
		printf("\n\nFinding mean of powmatrix...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		mean_of_matrix(rowVect, matrix, ROWS, COLS);
		printf("meanmatrix = \n");
		print_matrix(rowVect, 1, COLS);
		printf("\n");
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  divide by constant test */
		printf("\n\nDividing powmatrix by 100...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n / 100\n");
		divide_by_constant(matrix, matrix, ROWS, COLS, 100);
		printf("quotmatrix = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  sum scalar test */
		printf("\n\nAdding 100 to quotmatrix...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n + 100\n");
		sum_scalar_matrix(matrix, matrix, ROWS, COLS, 100);
		printf("summatrix = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  multiply matrices test  */
		printf("\n\nTesting matrix multiplication...\n\n");
		printf("\tTest 1 of 3 (summatrix * Tmatrix)...\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n*\n");
		printf("\n");   print_matrix(Tmatrix, COLS, ROWS);   printf("\n\n");
		multiply_matrices(rowsXrows, matrix, Tmatrix, ROWS, ROWS, COLS);
		printf("rowsXrows = \n");
		print_matrix(rowsXrows, ROWS, ROWS);

		printf("\n\tTest 2 of 3 (Tmatrix * summatrix)...\n");
		printf("\n");   print_matrix(Tmatrix, COLS, ROWS);   printf("\n*\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		multiply_matrices(colsXcols, Tmatrix, matrix, COLS, COLS, ROWS);
		printf("colsXcols = \n");
		print_matrix(colsXcols, COLS, COLS);

		printf("\n\tTest 3 of 3 (colVect * rowVect)...\n");
		printf("\n");   print_matrix(colVect, ROWS, 1);   printf("\n*\n");
		printf("\n");   print_matrix(rowVect, 1, COLS);   printf("\n\n");
		multiply_matrices(matrix, colVect, rowVect, ROWS, COLS, 1);
		printf("matrix = \n");
		print_matrix(matrix, ROWS, COLS);

		if (ROWS == COLS) {
			printf("\n\tOptional test for square matrices (rowVect * colVect)...\n");
			printf("\n");   print_matrix(rowVect, 1, COLS);   printf("\n*\n");
			printf("\n");   print_matrix(colVect, ROWS, 1);   printf("\n\n");
			multiply_matrices(one, rowVect, colVect, 1, 1, ROWS);
			printf("one = \n");
			print_matrix(one, 1, 1);
		}
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  subtract matrices test  */
		printf("\n\nTesting subtract matrices (matrix - matrix)...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n-\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		subtract_matrices(matrix, matrix, matrix, ROWS, COLS);
		printf("difference = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  normalize test  */
		printf("\n\nTesting normalization...\n\n");
		printf("\n");   print_matrix(Tmatrix, COLS, ROWS);   printf("\n\n");
		normalize(Tmatrix, Tmatrix, COLS, ROWS);
		printf("normalize = \n");
		print_matrix(Tmatrix, COLS, ROWS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  ones testing    */
		printf("\n\nTesting ones matrix function...\n\n");
		ones(matrix, ROWS, COLS);
		printf("ones = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  scale matrix testing    */
		printf("\n\nTesting scale matrix function...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		scale_matrix(matrix, matrix, ROWS, COLS, 10);
		printf("scale = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  Identity Matrix Test    */
		if (ROWS == COLS) {
			printf("\n\nTesting Identity function...\n\n");
			eye(matrix, ROWS);
			printf("eye = \n");
			print_matrix(matrix, ROWS, COLS);
			printf("Press any key to continue...\n");
			getchar();
			system("clear");
		}
		
		/*  flip matrix test    */
		printf("\n\nTesting fliplr function...\n\n");
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		fliplr(matrix, matrix, ROWS, COLS);
		printf("fliplr = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  acos test   */
		printf("\n\nTesting matrix_acos function...\n\n");
		printf("\n");   print_matrix(Tmatrix, COLS, ROWS);   printf("\n\n");
		matrix_acos(Tmatrix, Tmatrix, COLS, ROWS);
		printf("acosMat =\n");
		print_matrix(Tmatrix, COLS, ROWS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  reorder matrix test */
		printf("\n\nTesting reorder matrix function...\n\n");
		for (i = 0; i < ROWS; i++) {
			for (j = 0; j < COLS; j++) {
				matrix[i * COLS + j] = (data_t)(i * ROWS + j);
				rowVect[j] = (data_t)(COLS - 1 - j);
			}
		}
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n~\n");
		printf("\n");   print_matrix(rowVect, 1, COLS);   printf("\n\n");
		reorder_matrix(matrix, matrix, ROWS, COLS, rowVect);
		printf("reorderMat = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  fix function test   */
		printf("\n\nTesting fix function...\n\n");
		divide_by_constant(matrix, matrix, ROWS, COLS, 10);
		sum_scalar_matrix(matrix, matrix, ROWS, COLS, -1);
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n\n");
		fix(matrix, matrix, ROWS, COLS);
		printf("fixedMat = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  dot division test   */
		printf("\n\nTesting matrix dot division function...\n\n");
		data_t *temp;
		allocate_matrix(&temp, ROWS, COLS);
		for (i = 0; i < ROWS; i++) {
			for (j = 0; j < COLS; j++) {
				temp[i * COLS + j] = (data_t) i + 1;
				matrix[i * COLS + j] = (data_t) j + 1;
			}
		}
		printf("\n");   print_matrix(matrix, ROWS, COLS);   printf("\n./\n");
		printf("\n");   print_matrix(temp, ROWS, COLS);   printf("\n\n");
		matrix_dot_division(matrix, matrix, temp, ROWS, COLS);
		printf("quotMat = \n");
		print_matrix(matrix, ROWS, COLS);
		printf("Press any key to continue...\n");
		getchar();
		system("clear");
		
		/*  cofactor test */
		if (ROWS == COLS) {
			data_t test[] = {1, 2, 3, 1, 2, 1, 3, 2, 1};
			data_t test2[9];
			printf("\n\nTesting cofactor function...\n\n");
			printf("\n");   print_matrix(test, 3, 3);   printf("\n\n");
			cofactor(test2, test, ROWS);
			printf("cofactorMat = \n");
			print_matrix(test2, 3, 3);
			printf("Press any key to continue...\n");
			getchar();
			system("clear");

			/*  determinant test    */
			printf("\n\nTesting determinant function...\n\n");
			determinant(test, ROWS, &deter);
			printf("determinant = %f\n", deter);
			printf("Press any key to continue...\n");
			getchar();
			system("clear");
			
			/*  inverse test    */
			printf("\n\nTesting inverse function...\n\n");
			printf("\n");   print_matrix(test, 3, 3);   printf("\n");
			inv(test2, test, 3);
			printf("invMat = \n");
			print_matrix(test2, 3, 3);
			getchar();
			system("clear");
			
			printf("\n\nTesting Covariance function..\n\n");
			printf("\n"); print_matrix(test, 3, 3); printf("\n");
			covariance(test2, test, 3, 3);
			printf("covarinace matrix = \n");
			print_matrix(test2, 3, 3);
			getchar();
			system("clear");
			/*
			printf("\n\nTesting matrix sqrtm function..\n\n");
			printf("\n"); print_matrix(test, 3, 3); printf("\n");
			matrix_sqrtm(test2, test, 3, 3);
			printf("sqrtm matrix = \n");
			print_matrix(test2, 3, 3);
			getchar();
			system("clear");
			*/
			// matrix_eig is used by matrix sqrtm and is thus proved correct by correct matrix sqrtm function		
		}
		
		
		printf("\n\nTesting matrix exp function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n");
		matrix_exp(matrix, matrix, ROWS, COLS);
		printf("exp matrix = \n");
		print_matrix(matrix, ROWS, COLS);
		getchar();
		system("clear");
		
		
		printf("\n\nTesting mean_of_matrix function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n");
		mean_of_matrix_by_rows(matrix, matrix, ROWS, COLS);
		printf("exp matrix = \n");
		print_matrix(matrix, ROWS, COLS);
		getchar();
		system("clear");
	
		printf("\n\nTesting add_matrices function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n \t + \t\n"); print_matrix(matrix, ROWS, COLS);    printf("\n");
		add_matrices(matrix_out, matrix, matrix, ROWS, COLS);
		printf("sum = \n");
		print_matrix(matrix_out, ROWS, COLS);
		getchar();
		system("clear");
		
		printf("\n\nTesting matrix negate function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n");
		matrix_negate(matrix_out, matrix, ROWS, COLS);
		printf("matrix_negate = \n");
		print_matrix(matrix_out, ROWS, COLS);
		getchar();
		system("clear");
	}	
	
		printf("\n\nTesting matrix eig function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n");
		data_t *outvect, *outeigvals;
		allocate_matrix(&outvect, ROWS, COLS);
	    allocate_matrix(&outeigvals, ROWS, COLS);
		matrix_eig(outvect, outeigvals, matrix, ROWS, COLS);
		printf("eig vectors matrix = \n");
		print_matrix(outvect, ROWS, COLS);
		printf("eig vals matrix = \n");
		print_matrix(outeigvals, ROWS, COLS);
		getchar();
		system("clear");
		
	
		printf("\n\nTesting matrix sqrtm function..\n\n");
		printf("\n"); print_matrix(matrix, ROWS, COLS); printf("\n");
		matrix_sqrtm(matrix, matrix_out, ROWS, COLS);
		printf("sqrtm matrix = \n");
		print_matrix(matrix_out, ROWS, COLS);
		getchar();
		system("clear");

	
	/*  free matrices   */
    printf("\n\nFreeing memory...\n\n");
    free_matrix(&matrix);
    free_matrix(&Tmatrix);
    free_matrix(&colVect);
    free_matrix(&rowVect);
    free_matrix(&rowsXrows);
    free_matrix(&colsXcols);
    free_matrix(&one);
    
    printf("Completed Test!!\n");
    
    return 0;
}
