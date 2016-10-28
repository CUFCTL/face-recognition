/**
 * @file ica.c
 *
 * Implementation of ICA (Hyvarinen, 1999).
 *
 * Random normal distribution code from http://www.csee.usf.edu/~kchriste/tools/gennorm.c
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"
#include "matrix.h"

#define maxNumIterations 100
#define epsilon 0.0001

matrix_t * fpica (matrix_t * X, matrix_t * dewhiteningMatrix, matrix_t * whiteningMatrix);
void temp_PCA(matrix_t * X, matrix_t ** L_eval, matrix_t **L_evec);

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
    m_elem_apply(D, sqrt);
    matrix_t *inv_sqrt_D = m_inverse(D);
    matrix_t *E_tr = m_transpose(E);
    *whiteningMatrix = m_product(inv_sqrt_D, E_tr);
    *dewhiteningMatrix = m_product(E, D);

    matrix_t * newVectors = m_product(*whiteningMatrix, m_transpose(X));

    // cleanup
    m_free(inv_sqrt_D);
    m_free(E_tr);

    return newVectors;
}

// L_eval.... eigenvalue vector... must translate this to diagonal matrix
// L_evec.... eigenvector matrix
matrix_t * ICA (matrix_t *X, matrix_t *mean_face, matrix_t *L_eval, matrix_t *L_evec)
{
	// subtract mean "row" from X
	matrix_t * X_tr = m_transpose(X);
	matrix_t * mixedmean = m_mean_column(X_tr);

	m_subtract_columns(X_tr, mixedmean);

	// compute principal components
    matrix_t * e_vals = m_zeros(X->cols, 1);
    matrix_t * e_vecs = m_zeros(X->cols, X->cols);

    temp_PCA(X_tr, &e_vals, &e_vecs);
    matrix_t * D = m_diagonalize(e_vals);

    // call spherex after diagonalizing the eigenvalues
    matrix_t * whiteningMatrix, * dewhiteningMatrix;

    matrix_t * whitesig = sphere(X, e_vecs, D, &whiteningMatrix, &dewhiteningMatrix);

    // call fpica(whitened_matrix)
    // A is the mixing matrix
    matrix_t *W = fpica(whitesig, dewhiteningMatrix, whiteningMatrix);
    matrix_t *icasig = m_product(W, X_tr);
    matrix_t *icasig_temp1 = m_product(W, mixedmean);
    matrix_t *icasig_temp2 = m_ones(1, X_tr->cols);
    matrix_t *icasig_temp3 = m_product(icasig_temp1, icasig_temp2);

	m_add(icasig, icasig_temp3);

    // cleanup
    m_free(D);
    m_free(whiteningMatrix);
    m_free(dewhiteningMatrix);
    m_free(whitesig);

    return icasig;
}


void temp_PCA(matrix_t * X, matrix_t ** L_eval, matrix_t **L_evec)
{
    matrix_t * cov_m = m_covariance(X);
    m_eigen(cov_m, *L_eval, *L_evec);
}


/**
 * Function    : fpica
 * Parameters  : X   -> whitened data as column vectors
 * Return      : A   -> mixing matrix (W is A's inverse)
 * Purpose     : This function runs the deflation ICA method from the Matlab
 *               implementation provided by http://research.ics.aalto.fi/ica/fastica/ .
 *               whitesig must be input as a whitened data matrix with the input
 *               and whitened/mean subtracted/PCA-performed preprocessing steps
 *               already complete. The function will return the mixing matrix A
 *               which contain the ICA components.
 *               We are implementing the deflation approach and using the nonlinearity
 *               function pow3.
 */

// this translation starts at line 582 of fpica.m
matrix_t * fpica (matrix_t * X, matrix_t * dewhiteningMatrix, matrix_t * whiteningMatrix)
{
    int round = 1, c = 0;

    int numSamples = X->cols; // make sure this and vectorSize are correct
    int vectorSize = X->rows;

    // initialize output matrices
    matrix_t * B = m_zeros(vectorSize, vectorSize);
    //matrix_t * A = m_zeros(vectorSize, vectorSize);
    matrix_t * W = m_zeros(vectorSize, vectorSize);

    // initialize matrices
    matrix_t * wOld = m_zeros(vectorSize, 1);

    while (round <= vectorSize)
    {
        // BEGIN line 613 fpica.m
        matrix_t * w = m_random(vectorSize, 1);

        // helper matrices for line 613 of fpica.m
        matrix_t * transposeB = m_transpose(B);
        matrix_t * tempB = m_product(B, transposeB);
        matrix_t * tempB_prod_w = m_product(tempB, w);

        m_subtract(w, tempB_prod_w);

        m_elem_mult(w, (1/m_norm(w)));

        m_free(transposeB);
        m_free(tempB);
        m_free(tempB_prod_w);
        // END line 613 fpica.m

        printf("round %d\n", round);

        int i = 0;

        while (i <= maxNumIterations + 1)
        {
            // Project the vector into the space orthogonal to the space
            // spanned by the earlier found basis vectors. Note that we can do
            // the projection with matrix B, since the zero entries do not
            // contribute to the projection.
            transposeB = m_transpose(B);
            tempB = m_product(B, transposeB);
            tempB_prod_w = m_product(tempB, w);

            m_subtract(w, tempB_prod_w);
            m_elem_mult(w, (1/m_norm(w)));

            matrix_t * copy_w = m_copy(w);
            matrix_t * copy_w2 = m_copy(w);

            m_subtract(copy_w, wOld);
            m_add(copy_w2, wOld);

            // line 680 of fpica.m
            if (m_norm(copy_w) < epsilon || m_norm(copy_w2) < epsilon)
            {
                // save the vector w to the matrix B
                for (c = 0; c < vectorSize; c++)
                {
                    elem(B, c, round) = elem(w, c, 0);
                }

                // calculate the de-whitened vector
                //matrix_t * temp_A_vec = m_product(dewhiteningMatrix, w);

                // save the de-whitened vector to matrix A
                //for (c = 0; c < vectorSize; c++)
                //{
                //    elem(A, c, round) = elem(temp_A_vec, c, 0);
                //}

                //m_free(temp_A_vec);

                // calculate ICA filter
                matrix_t * temp_W_vec = m_product(m_transpose(w), whiteningMatrix);

                // save the ICA filter vector
                for (c = 0; c < vectorSize; c++)
                {
                    elem(W, round, c) = elem(temp_W_vec, 0, c);
                }

                m_free(temp_W_vec);
            }

            wOld = w;

            // pow3 function on w on line 767 of fpica.m
            matrix_t * X_tran = m_transpose(X);
            matrix_t * X_tran_w = m_product(X_tran, w);
            m_free(X_tran);

            // raise each element to the third power
            for (c = 0; c < vectorSize; c++)
            {
                elem(X_tran_w, c, 0) = pow(elem(X_tran_w, c, 0), 3);
            }

            matrix_t * w_cpy = m_copy(w);
            m_elem_mult(w_cpy, 3);
            w = m_product(X, X_tran_w);
            m_elem_mult(w, (1/numSamples));
            m_subtract(w, w_cpy);

            m_free(w_cpy);
            m_free(X_tran_w);

            m_free(copy_w);
            m_free(copy_w2);
            m_free(transposeB);
            m_free(tempB);
            m_free(tempB_prod_w);

            m_elem_mult(w, (1/m_norm(w)));
            i++;
        }

        m_free(w);
        round++;
    }


    return W;
}
