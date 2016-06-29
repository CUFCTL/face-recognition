/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"

/**
 * Apply a "sphere" transformation to a vector.
 *
 * @param x  pointer to column vector
 * @return pointer to the transformed vector
 */
matrix_t * sphere(matrix_t *x)
{
    // TODO: subtract mean from x

    // compute W_z = 2 * Cov(X)^(-1/2)
    matrix_t *W_z_temp1 = m_covariance(x);
    matrix_t *W_z_temp2 = m_sqrtm(W_z_temp1);
    matrix_t *W_z = m_inverse(W_z_temp2);
    m_elem_mult(W_z, 2);

    // compute x_out = W_z * x
    matrix_t *x_out = m_product(W_z, x);

    free(W_z_temp1);
    free(W_z_temp2);
    free(W_z);

    return x_out;
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
