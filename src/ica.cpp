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

matrix_t * fpica (matrix_t *X, matrix_t *whiteningMatrix);

/**
 * An alternate implementation of PCA.
 *
 * This implementation conforms to pcamat.m in the MATLAB ICA code;
 * however, it produces eigenvectors with different dimensions from
 * our C implementation of PCA. Until these two functions can be resolved,
 * ICA will have to use this function.
 *
 * @param X    input matrix in columns
 * @param p_D  pointer to store eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA_alt(matrix_t *X, matrix_t **p_D)
{
    // compute the covariance of X
    matrix_t *C = m_covariance(X);

    // compute the eigenvalues, eigenvectors of the covariance
    matrix_t *E_temp1;
    matrix_t *D_temp1;

    m_eigen(C, &E_temp1, &D_temp1);

    // take the highest N-1 eigenvalues
    matrix_t *E = m_copy_columns(E_temp1, 1, E_temp1->cols);
    matrix_t *D_temp2 = m_copy_columns(D_temp1, 1, D_temp1->cols);
    matrix_t *D = m_copy_rows(D_temp2, 1, D_temp2->rows);

    // save outputs
    *p_D = D;

    // cleanup
    m_free(C);
    m_free(E_temp1);
    m_free(D_temp1);
    m_free(D_temp2);

    return E;
}

/**
 * Compute the whitening matrix for a matrix X.
 *
 * The whitening matrix, when applied to X, removes
 * the first- and second-order statistics; that is,
 * the mean and covariances are set to zero and the
 * variances are equalized.
 *
 * @param X  input matrix
 * @param E  matrix of eigenvectors in columns
 * @param D  diagonal matrix of eigenvalues
 * @param p_whiteningMatrix
 * @return whitened output matrix
 */
matrix_t * whiten (matrix_t *X, matrix_t *E, matrix_t *D, matrix_t **p_whiteningMatrix)
{
    // compute whitening matrix
    matrix_t *D_temp1 = m_copy(D);
    m_elem_apply(D_temp1, sqrtf);

    matrix_t *D_temp2 = m_inverse(D_temp1);
    matrix_t *E_tr = m_transpose(E);
    matrix_t *whiteningMatrix = m_product(D_temp2, E_tr);

    // compute output matrix
    matrix_t *whitesig = m_product(whiteningMatrix, X);

    // cleanup
    m_free(D_temp1);
    m_free(D_temp2);
    m_free(E_tr);

    *p_whiteningMatrix = whiteningMatrix;

    return whitesig;
}

/**
 * Compute the independent components of a matrix of image vectors.
 *
 * TODO: should we not subtract the mean column from X beforehand?
 *
 * @param X  matrix of mean-subtracted images in columns
 * @return independent components of X in columns
 */
matrix_t * ICA (matrix_t *X)
{
    // compute mixedsig = X', subtract mean column
    matrix_t *mixedsig = m_transpose(X);
    matrix_t *mixedmean = m_mean_column(mixedsig);

    m_subtract_columns(mixedsig, mixedmean);

    // compute principal components
    matrix_t *D;
    matrix_t *E = PCA_alt(X, &D);

    // compute whitened input
    matrix_t *whiteningMatrix;
    matrix_t *whitesig = whiten(mixedsig, E, D, &whiteningMatrix);

    // compute mixing matrix
    matrix_t *W = fpica(whitesig, whiteningMatrix);

    // compute independent components
    // icasig = W * mixedsig + (W * mixedmean) * ones(1, mixedsig->cols)
    matrix_t *icasig = m_product(W, mixedsig);
    matrix_t *icasig_temp1 = m_product(W, mixedmean);
    matrix_t *icasig_temp2 = m_ones(1, mixedsig->cols);
    matrix_t *icasig_temp3 = m_product(icasig_temp1, icasig_temp2);

    m_add(icasig, icasig_temp3);

    // cleanup
    m_free(mixedsig);
    m_free(mixedmean);
    m_free(E);
    m_free(D);
    m_free(whiteningMatrix);
    m_free(whitesig);
    m_free(W);
    m_free(icasig_temp1);
    m_free(icasig_temp2);
    m_free(icasig_temp3);

    return icasig;
}

/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 * @return x ^ 3
 */
precision_t pow3(precision_t x)
{
    return pow(x, 3);
}

/**
 * Compute the mixing matrix W for an input matrix X using the deflation
 * approach and the nonlinearity functions pow3. The input matrix should
 * already be whitened.
 *
 * @param X
 * @param whiteningMatrix
 * @return mixing matrix W
 */
matrix_t * fpica (matrix_t *X, matrix_t *whiteningMatrix)
{
    const int MAX_ITERATIONS = 1000;
    const precision_t EPSILON = 0.0001;

    int vectorSize = X->rows;
    int numSamples = X->cols;

    matrix_t *B = m_zeros(vectorSize, vectorSize);
    matrix_t *W = m_zeros(vectorSize, whiteningMatrix->cols);
    matrix_t *X_tr = m_transpose(X);

    int i;
    for ( i = 0; i < vectorSize; i++ ) {
        if ( VERBOSE ) {
            printf("round %d\n", i + 1);
        }

        // initialize w as a Gaussian random vector
        matrix_t *w = m_random(vectorSize, 1);

        // compute w = (w - B * B' * w), normalize w
        matrix_t *B_tr = m_transpose(B);
        matrix_t *w_temp1 = m_product(B, B_tr);
        matrix_t *w_temp2 = m_product(w_temp1, w);

        m_subtract(w, w_temp2);
        m_elem_mult(w, 1 / m_norm(w));

        m_free(B_tr);
        m_free(w_temp1);
        m_free(w_temp2);

        // initialize w0
        matrix_t *w0 = m_zeros(w->rows, w->cols);

        int j;
        for ( j = 0; j < MAX_ITERATIONS; j++ ) {
            // compute w = (w - B * B' * w), normalize w
            B_tr = m_transpose(B);
            w_temp1 = m_product(B, B_tr);
            w_temp2 = m_product(w_temp1, w);

            m_subtract(w, w_temp2);
            m_elem_mult(w, 1 / m_norm(w));

            m_free(B_tr);
            m_free(w_temp1);
            m_free(w_temp2);

            // compute w_delta1 = w - w0
            matrix_t *w_delta1 = m_copy(w);
            m_subtract(w_delta1, w0);

            // compute w_delta2 = w + w0
            matrix_t *w_delta2 = m_copy(w);
            m_add(w_delta2, w0);

            // determine whether the direction of w and w0 are equal
            precision_t norm1 = m_norm(w_delta1);
            precision_t norm2 = m_norm(w_delta2);

            if ( VERBOSE ) {
                printf("%lf %lf\n", norm1, norm2);
            }

            int converged = (norm1 < EPSILON) || (norm2 < EPSILON);

            m_free(w_delta1);
            m_free(w_delta2);

            // terminate round if w converges
            if ( converged ) {
                // save B(:, i) = w
                m_assign_column(B, i, w, 0);

                // save W(i, :) = w' * whiteningMatrix
                matrix_t *w_tr = m_transpose(w);
                matrix_t *W_temp1 = m_product(w_tr, whiteningMatrix);

                m_assign_row(W, i, W_temp1, 0);

                // cleanup
                m_free(w_tr);
                m_free(W_temp1);

                // continue to the next round
                break;
            }

            // update w0
            m_assign_column(w0, 0, w, 0);

            // compute w = X * ((X' * w) .^ 3) / numSamples - 3 * w
            w_temp1 = m_product(X_tr, w);
            m_elem_apply(w_temp1, pow3);

            w_temp2 = m_copy(w);
            m_elem_mult(w_temp2, 3);

            w = m_product(X, w_temp1);
            m_elem_mult(w, 1.0 / numSamples);
            m_subtract(w, w_temp2);

            // normalize w
            m_elem_mult(w, 1 / m_norm(w));

            m_free(w_temp1);
            m_free(w_temp2);
        }

        if ( VERBOSE ) {
            printf("iterations: %d\n", j);
        }

        m_free(w);
        m_free(w0);
    }

    m_free(B);
    m_free(X_tr);

    return W;
}
