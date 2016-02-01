/*==================================================================================================
 *  testFunctions.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *
 *  This file contains
 *		main
 *  
 *  Lasted Edited: Jun. 26, 2013
 *
 *  Changes made: by William - created this file
 *
 */
 
#include "levelTwoOps.c"
#include "pcabigFn.c"
#include "runicaAux.c"

#define ROWS1 5
#define COLS1 5
#define ROWS2 5
#define COLS2 5
 
/*==================================================================================================
 *  main
 *
 *  Parameters
 *
 *  Returns
 *
 *  Description: The purpose of this function is to test higher level auxiliary functions that
 *  have been completed (ie. cosFn and zero_mean).
 *
 *  THIS FUNCTION CALLS
 *
 */
int main() {
    int i, j;
    data_t *cosOut, *cosIn1, *cosIn2, *zeroMeanOut, *zeroMeanIn;
    
    /*  allocating Matrices */
    allocate_matrix(&cosOut, COLS1, COLS2);
    allocate_matrix(&cosIn1, ROWS1, COLS1);
    allocate_matrix(&cosIn2, ROWS2, COLS2);
    allocate_matrix(&zeroMeanOut, ROWS1, COLS1);
    allocate_matrix(&zeroMeanIn, ROWS1, COLS1);
    /*  end allocation  */
    
    /*  testing cosfn   */
    for (i = 0; i < ROWS1; i++) {
        for (j = 0; j < COLS1; j++) {
            cosIn1[i * COLS1 + j] = (data_t)(i * ROWS1 + j);
        }
    }
    for (i = 0; i < ROWS2; i++) {
        for (j = 0; j < COLS2; j++) {
            //cosIn2[i * COLS2 + j] = (data_t)((ROWS2 * COLS2) - (i * ROWS1 + j));
            cosIn2[i * COLS2 + j] = (data_t)(i * ROWS1 + j);
            //cosIn2[i * COLS2 + j] = (data_t)7;
        }
    }
    printf("\n\nTesting cosFn...\n\n");
    printf("\n");   print_matrix(cosIn1, ROWS1, COLS1);   printf("\n\n");
    printf("\n");   print_matrix(cosIn2, ROWS2, COLS2);   printf("\n\n");
    cosFn(cosOut, cosIn1, cosIn2, ROWS1, COLS1, ROWS2, COLS2);
    printf("cosFnMat = \n");
    print_matrix(cosOut, ROWS1, COLS1);
    printf("Press any key to continue...\n");
    getchar();
    system("clear");
    
    /*  testing zero mean   */
    printf("\n\nTesting zero_mean...\n\n");
    for (i = 0; i < ROWS1; i++) {
        for (j = 0; j < COLS1; j++) {
            zeroMeanIn[i * COLS1 + j] = (data_t)(i * ROWS1 + j);
        }
    }
    printf("\n");   print_matrix(zeroMeanIn, ROWS1, COLS1);   printf("\n\n");
    zero_mean(zeroMeanOut, zeroMeanIn, ROWS1, COLS1);
    printf("zeroMeanMat = \n");
    print_matrix(zeroMeanOut, ROWS1, COLS1);
    printf("Press any key to continue...\n");
    getchar();
    system("clear");
    
    /*  Testing mergesort function  */
    printf("\n\nTesting Mergesort function...\n\n");
    free_matrix(&cosOut);
    allocate_matrix(&cosOut, 1, COLS1);
    for (i = 0; i < COLS1; i++) cosOut[i] = (double)(COLS1 - i);
    printf("\n");   print_matrix(cosIn1, ROWS1, COLS1);   printf("\n\n");
    printf("\n");   print_matrix(cosOut, 1, COLS1);   printf("\n\n");
    eigSort(cosIn1, cosOut, COLS1, ROWS1);
    printf("sortedMat = \n");
    print_matrix(cosIn1, ROWS1, COLS1);   printf("\n\n");
    printf("\n");   print_matrix(cosOut, 1, COLS1);   printf("\n\n");
    
    /*  testing pcaBigFunction  */
    printf("\n\nTesting pcabigFn...\n\n");
    printf("Initializing variables...\n");
    printf("\tmatrix = \n\t");
    print_matrix(zeroMeanIn, ROWS1, COLS1);
    printf("\n\trows = %d\n\tcols = %d\n", ROWS1, COLS1);
    printf("\tmatrix_length = \n");
//    print_matrix();
    printf("\nCalling pcabigFn...\n\tvoid pcabigFn(data_t **matrix, int rows, int cols, data_t **matrix_length)\n\n");
    //pcabigFn(data_t **U, data_t **R, data_t **E, int rows, int cols, data_t **B, int rows, int cols)
	pcabigFn(data_t **U, data_t **R, data_t **E, int rows, int cols, data_t **B, int rows, int cols)
	
	
	/*   testing spherex  */
	printf("\n\nTesting Spherex... \n\n");
	//spherex(data_t *x, data_t *oldx, int rows, int cols, double P, data_t *wz);
	
	
	
	
    /*  freeing memory  */
    printf("\n\nFreeing allocated memory...\n\n");
    free_matrix(&cosOut);
    free_matrix(&cosIn1);
    free_matrix(&cosIn2);
    free_matrix(&zeroMeanOut);
    free_matrix(&zeroMeanIn);
    
    printf("Test Completed!!\n");
    
    return 0;
}