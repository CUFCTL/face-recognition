/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>

// THIS FILE SHOULD PREPROCESS THE DATA THAT WILL BE CALLED USING THE FPICA ALGORITHM
//  * WHITEN THE DATA
//  * CALCULATE PCA

/**
 * Compute the whitening matrix W_z for a matrix X.
 *
 * The whitening matrix, when applied to X, removes
 * the first- and second-order statistics; that is,
 * the mean and covariances are set to zero and the
 * variances are equalized.
 *
 * @param X  mean-subtracted input matrix
 * @return whitening matrix W_z
 */
matrix_t * sphere(matrix_t *X)
{
    // compute W_z = 2 * Cov(X)^(-1/2)
    matrix_t *W_z_temp1 = m_covariance(X);
    matrix_t *W_z_temp2 = m_sqrtm(W_z_temp1);
    matrix_t *W_z = m_inverse(W_z_temp2);
    m_elem_mult(W_z, 2);

    // cleanup
    m_free(W_z_temp1);
    m_free(W_z_temp2);

    return W_z;
}

// L_eval.... eigenvalue vector... must translate this to diagonal matrix
// L_evec.... eigenvector matrix
matrix_t * ICA(matrix_t *X, matrix_t *L_eval, matrix_t *L_evec)
{
    // call spherex
    // call fpica(whitened_matrix)
}
