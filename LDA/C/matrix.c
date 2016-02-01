#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

/*
 * Allocates and returns a MATRIX * object
 * rows, cols: number of rows, columns to be allocated
 */
MATRIX * matrix_constructor(int rows, int cols)
{
    int i;
    MATRIX * M = (MATRIX *) malloc(sizeof(MATRIX));
    double ** data = (double **) malloc(rows * sizeof(double *));
    double * datap = (double *) malloc(rows * cols * sizeof(double));
    M->rows = rows;
    M->cols = cols;

    //set each data pointer to the first element in each row
    for (i = 0; i < rows; i++) {
        data[i] = &datap[i * cols];
    }

    M->data = data;

    return M;
}

/*
 * Print function
 */
void matrix_print(MATRIX *M, int decimals)
{
    int i, j;

    for(i = 0; i < M->rows; i++) {
        for(j = 0; j < M->cols; j++) {
            printf("%12.*lf", decimals, M->data[i][j]);
        }
        printf("\n");
    }
}

/*
 * Finds the mean between each of the columns of the matrix
 * A: m x n matrix
 * returns: m x 1 matrix
 */
MATRIX * matrix_mean(MATRIX * A)
{
    int rows = A->rows;
    int cols = A->cols;
    int i, j;
    double temp;

    MATRIX * B = matrix_constructor(rows, 1);

    for (i = 0; i < rows; i++) {
        temp = 0;
        for (j = 0; j < cols; j++) {
            temp += (double) A->data[i][j];
        }
        B->data[i][0] = (temp / cols);
    }

    return B;
}

/*
 * Frees the MATRIX * object
 * M: the MATRIX * to be destroyed
 */
void matrix_destructor(MATRIX * M)
{
    free(*M->data);
    free(M->data);
    free(M);
    return;
}


/*matrix_bounded_mean
 * Arguments:   A           Matrix containing data to be averaged
 *              start_row   index of first row to be summed
 *              end_row     index of last row to be summed
 *              start_col   index of first column in sum
 *              end_col     index of last column within sum
 * Returns:     MATRIX type containing a column vector which is the matrix
 *              mean of the sub matrix referenced by the arguments of the func
 */
MATRIX *matrix_bounded_mean(MATRIX *A, int start_row, int end_row, int start_col, int end_col)
{
    int i, j;
    int sum;

    MATRIX * B = matrix_constructor(end_row - start_row + 1, 1);

    for (i = start_row; i <= end_row, i++) {
        sum = 0;
        for (j = start_col; j <= end_col; j++) {
            sum += (double) A->data[i][j];
        }
        B->data[i][0] = (sum / (end_col - start_col + 1));
    }

    return B;
}
