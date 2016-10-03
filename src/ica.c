/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>

matrix_t * diagonalize(matrix_t *M);
void m_elem_sqrt (matrix_t * M);

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
matrix_t * sphere (matrix_t *X, matrix_t *E, matrix_t *D, matrix_t **whiteningMatrix, matrix_t **dewhiteningMatrix)
{
    // compute whitened data based on whitenv.m
    m_elem_sqrt(D);
    matrix_t *inv_sqrt_D = m_inverse(D);
    matrix_t *E_tr = m_transpose(E);
    *whiteningMatrix = m_product(inv_sqrt_D, E_tr);
    *dewhiteningMatrix = m_product(E, D);

    matrix_t * newVectors = m_product(whiteningMatrix, X);

    // cleanup
    m_free(inv_sqrt_D);
    m_free(E_tr);

    return newVectors;
}

// L_eval.... eigenvalue vector... must translate this to diagonal matrix
// L_evec.... eigenvector matrix
matrix_t * ICA (matrix_t *X, matrix_t *L_eval, matrix_t *L_evec)
{
    // call spherex after diagonalizing the eigenvalues
    matrix_t * whiteningMatrix, * dewhiteningMatrix;
    matrix_t * D = diagonalize(L_eval);
    matrix_t * whitesig = sphere(X, L_evec, D, &whiteningMatrix, &dewhiteningMatrix);

    // call fpica(whitened_matrix)


    // cleanup
    m_free(D);
    m_free(whiteningMatrix);
    m_free(dewhiteningMatrix);
    m_free(whitesig);

    return NULL;
}

matrix_t * diagonalize (matrix_t *M)
{
    int i, j;

    matrix_t * D = m_zeroes(M->rows, M->rows);

    for (i = 0; i < M->rows; i++)
    {
        elem(D, i, i) = elem(M, i, 0);
    }

    return D;
}

void m_elem_sqrt (matrix_t * M)
{
    int i, j;

    for (i = 0; i < M->rows; i++)
    {
        for (j = 0; j < M->cols; j++)
        {
            elem(M, i , j) = sqrt(elem(M, i, j));
        }
    }
}
