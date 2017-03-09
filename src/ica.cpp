/**
 * @file ica.cpp
 *
 * Implementation of ICA (Hyvarinen, 1999).
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"
#include "timer.h"

matrix_t * fpica (matrix_t *X, matrix_t *W_z, int n2, int max_iterations, precision_t epsilon);

/**
 * Compute the whitening transformation for a matrix X.
 *
 * The whitening (sphering) matrix, when applied to X, transforms
 * X to have zero mean and unit covariance.
 *
 * @param X
 * @param n1
 * @return whitening matrix W_z
 */
matrix_t * m_whiten (matrix_t *X, int n1)
{
    // compute [V, D] = eig(C)
    matrix_t *D;
    matrix_t *V = PCA(X, n1, &D);

    // compute whitening matrix W_z = inv(sqrt(D)) * V'
    matrix_t *D_temp1 = m_copy("sqrt(D)", D);
    m_elem_apply(D_temp1, sqrtf);

    matrix_t *D_temp2 = m_inverse("inv(sqrt(D))", D_temp1);
    matrix_t *W_z = m_product("W_z", D_temp2, V, false, true);

    // cleanup
    m_free(V);
    m_free(D);
    m_free(D_temp1);
    m_free(D_temp2);

    return W_z;
}

/**
 * Compute the independent components of a matrix X, which
 * consists of observations in columns.
 *
 * NOTE: Curently, X is transposed before it is processed, which
 * causes there to be two extra transposes, an extra mean subtraction,
 * and an extra PCA calculation. We should try to refactor ICA to use
 * X in its original form to eliminate these redundancies.
 *
 * @param X
 * @param n1
 * @param n2
 * @param max_iterations
 * @param epsilon
 * @return independent components of X in columns
 */
matrix_t * ICA (matrix_t *X, int n1, int n2, int max_iterations, precision_t epsilon)
{
    timer_push("  ICA");

    timer_push("    subtract mean from input matrix");

    // compute mixedsig = X', subtract mean column
    matrix_t *mixedsig = m_transpose("mixedsig", X);
    matrix_t *mixedmean = m_mean_column("mixedmean", mixedsig);

    m_subtract_columns(mixedsig, mixedmean);

    timer_pop();

    timer_push("    compute whitening matrix and whitened input matrix");

    // compute whitened input U = W_z * mixedsig
    matrix_t *W_z = m_whiten(mixedsig, n1);
    matrix_t *U = m_product("U", W_z, mixedsig);

    timer_pop();

    timer_push("    compute mixing matrix");

    // compute mixing matrix
    matrix_t *W = fpica(U, W_z, n2, max_iterations, epsilon);

    timer_pop();

    timer_push("    compute ICA projection matrix");

    // compute independent components
    // icasig = W * (mixedsig + mixedmean * ones(1, mixedsig->cols))
    matrix_t *ones = m_ones("ones", 1, mixedsig->cols);
    matrix_t *icasig_temp1 = m_product("icasig_temp1", mixedmean, ones);

    m_add(icasig_temp1, mixedsig);

    matrix_t *icasig = m_product("icasig", W, icasig_temp1);

    // compute W_ica = icasig'
    matrix_t *W_ica = m_transpose("W_ica", icasig);

    timer_pop();

    timer_pop();

    // cleanup
    m_free(mixedsig);
    m_free(mixedmean);
    m_free(W_z);
    m_free(U);
    m_free(W);
    m_free(ones);
    m_free(icasig_temp1);
    m_free(icasig);

    return W_ica;
}

/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 * @return x^3
 */
precision_t pow3(precision_t x)
{
    return powf(x, 3);
}

/**
 * Compute the mixing matrix W for an input matrix X using the deflation
 * approach and the nonlinearity functions pow3. The input matrix should
 * already be whitened.
 *
 * @param X
 * @param W_z
 * @param n2
 * @param max_iterations
 * @param epsilon
 * @return mixing matrix W
 */
matrix_t * fpica (matrix_t *X, matrix_t *W_z, int n2, int max_iterations, precision_t epsilon)
{
    int vectorSize = X->rows;
    int numSamples = X->cols;

    // if n2 is -1, use default value
    n2 = (n2 == -1)
        ? vectorSize
        : n2;

    matrix_t *B = m_zeros("B", vectorSize, vectorSize);
    matrix_t *W = m_zeros("W", n2, W_z->cols);

    int i;
    for ( i = 0; i < n2; i++ ) {
        if ( LOGGER(LL_VERBOSE) ) {
            printf("      round %d\n", i + 1);
        }

        // initialize w as a Gaussian (0, 1) random vector
        matrix_t *w = m_random("w", vectorSize, 1);

        // compute w = (w - B * B' * w), normalize w
        matrix_t *w_temp1 = m_product("w_temp1", B, B, false, true);
        matrix_t *w_temp2 = m_product("w_temp2", w_temp1, w);

        m_subtract(w, w_temp2);
        m_elem_mult(w, 1 / m_norm(w));

        m_free(w_temp1);
        m_free(w_temp2);

        // initialize w0
        matrix_t *w0 = m_zeros("w0", w->rows, w->cols);

        int j;
        for ( j = 0; j < max_iterations; j++ ) {
            // compute w = (w - B * B' * w), normalize w
            w_temp1 = m_product("w_temp1", B, B, false, true);
            w_temp2 = m_product("w_temp2", w_temp1, w);

            m_subtract(w, w_temp2);
            m_elem_mult(w, 1 / m_norm(w));

            m_free(w_temp1);
            m_free(w_temp2);

            // compute w_delta1 = w - w0
            matrix_t *w_delta1 = m_copy("w_delta1", w);
            m_subtract(w_delta1, w0);

            // compute w_delta2 = w + w0
            matrix_t *w_delta2 = m_copy("w_delta2", w);
            m_add(w_delta2, w0);

            // determine whether the direction of w and w0 are equal
            precision_t norm1 = m_norm(w_delta1);
            precision_t norm2 = m_norm(w_delta2);

            m_free(w_delta1);
            m_free(w_delta2);

            // terminate round if w converges
            if ( norm1 < epsilon || norm2 < epsilon ) {
                // save B(:, i) = w
                m_assign_column(B, i, w, 0);

                // save W(i, :) = w' * W_z
                matrix_t *W_temp1 = m_product("W_temp1", w, W_z, true, false);

                m_assign_row(W, i, W_temp1, 0);

                // cleanup
                m_free(W_temp1);

                // continue to the next round
                break;
            }

            // update w0
            m_assign_column(w0, 0, w, 0);

            // compute w = X * ((X' * w) .^ 3) / numSamples - 3 * w
            w_temp1 = m_product("w_temp1", X, w, true, false);
            m_elem_apply(w_temp1, pow3);

            w_temp2 = m_copy("w_temp2", w);
            m_elem_mult(w_temp2, 3);

            w = m_product("w", X, w_temp1);
            m_elem_mult(w, 1.0 / numSamples);
            m_subtract(w, w_temp2);

            // normalize w
            m_elem_mult(w, 1 / m_norm(w));

            m_free(w_temp1);
            m_free(w_temp2);
        }

        if ( LOGGER(LL_VERBOSE) ) {
            printf("      iterations: %d\n", j);
        }

        m_free(w);
        m_free(w0);
    }

    m_free(B);

    return W;
}
