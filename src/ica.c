/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

/**
 * Compute the whitening matrix W_z for a matrix X.
 *
 * @param X  pointer to input matrix
 * @return pointer to whitening matrix W_z
 */
matrix_t * sphere(matrix_t *X)
{
    // compute mean-subtracted matrix A
    matrix_t *A = m_copy(X);
    matrix_t *m = m_mean_column(X);

    m_subtract_columns(A, m);

    // compute W_z = 2 * Cov(A)^(-1/2)
    matrix_t *W_z_temp1 = m_covariance(A);
    matrix_t *W_z_temp2 = m_sqrtm(W_z_temp1);
    matrix_t *W_z = m_inverse(W_z_temp2);
    m_elem_mult(W_z, 2);

    free(A);
    free(m);
    free(W_z_temp1);
    free(W_z_temp2);

    return W_z;
}

/**
 * Print stats for the change in weight matrix.
 *
 * @param W0  previous weight matrix
 * @param W   current weight matrix
 */
void sepout(matrix_t *W0, matrix_t *W)
{
    // compute the magnitude of W - W0
    precision_t norm = 0;

    int i, j;
    for ( i = 0; i < W->rows; i++ ) {
        for ( j = 0; j < W->cols; j++ ) {
            precision_t diff = elem(W, i, j) - elem(W0, i, j);
            norm += diff * diff;
        }
    }

    // compute the angle between W and W0
    // TODO: merge with m_dist_COS
    // compute W * W0
    precision_t x_dot_y = 0;

    for ( i = 0; i < W->rows; i++ ) {
        for ( j = 0; j < W->cols; j++ ) {
            x_dot_y += elem(W, i, j) * elem(W0, i, j);
        }
    }

    // compute ||W|| and ||W0||
    precision_t abs_x = 0;
    precision_t abs_y = 0;

    for ( i = 0; i < W->rows; i++ ) {
        for ( j = 0; j < W->cols; j++ ) {
            abs_x += elem(W, i, j) * elem(W, i, j);
            abs_y += elem(W0, i, j) * elem(W0, i, j);
        }
    }

    precision_t angle = acos(-x_dot_y / sqrt(abs_x * abs_y));

    printf("*** change: %.4lf angle: %.1lf deg\n", norm, 180 * angle / M_PI);
}

/**
 * Compute the projection matrix of a training set with ICA.
 *
 * @param A  mean-subtracted image matrix
 * @return projection matrix W_ica
 */
matrix_t * ICA(matrix_t *A)
{
    return NULL;
}
