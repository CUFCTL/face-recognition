/**
 * @file ica.cpp
 *
 * Implementation of ICA (Hyvarinen, 1999).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ica.h"
#include "logger.h"
#include "math_helper.h"
#include "pca.h"
#include "timer.h"

matrix_t * fpica (ica_params_t *params, matrix_t *X, matrix_t *W_z);

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
    pca_params_t pca_params = { n1 };

    // compute [V, D] = eig(C)
    matrix_t *D;
    matrix_t *V = PCA(&pca_params, X, &D);

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
 * @param params
 * @param X
 * @return independent components of X in columns
 */
matrix_t * ICA (ica_params_t *params, matrix_t *X)
{
    timer_push("ICA");

    timer_push("subtract mean from input matrix");

    // compute mixedsig = X', subtract mean column
    matrix_t *mixedsig = m_transpose("mixedsig", X);
    matrix_t *mixedmean = m_mean_column("mixedmean", mixedsig);

    m_subtract_columns(mixedsig, mixedmean);

    timer_pop();

    timer_push("compute whitening matrix and whitened input matrix");

    // compute whitened input U = W_z * mixedsig
    matrix_t *W_z = m_whiten(mixedsig, params->n1);
    matrix_t *U = m_product("U", W_z, mixedsig);

    timer_pop();

    timer_push("compute mixing matrix");

    // compute mixing matrix
    matrix_t *W = fpica(params, U, W_z);

    timer_pop();

    timer_push("compute ICA projection matrix");

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
 * Compute the parameter update for fpica
 * with the pow3 nonlinearity:
 *
 * g(u) = u^3
 * g'(u) = 3 * u^2
 *
 * which gives:
 *
 * w+ = X * ((X' * w) .^ 3) / X->cols - 3 * w
 * w* = w+ / ||w+||
 *
 * @param w0
 * @param X
 * @return w*
 */
matrix_t * fpica_pow3 (matrix_t *w0, matrix_t *X)
{
    // compute w+
    matrix_t *w_temp1 = m_product("w_temp1", X, w0, true, false);
    m_elem_apply(w_temp1, pow3);

    matrix_t *w_temp2 = m_copy("w_temp2", w0);
    m_elem_mult(w_temp2, 3);

    matrix_t *w = m_product("w", X, w_temp1);
    m_elem_mult(w, 1.0f / X->cols);
    m_subtract(w, w_temp2);

    // compute w*
    m_elem_mult(w, 1 / m_norm(w));

    // cleanup
    m_free(w_temp1);
    m_free(w_temp2);

    return w;
}

/**
 * Compute the parameter update for fpica
 * with the tanh nonlinearity:
 *
 * g(u) = tanh(u)
 * g'(u) = sech(u)^2 = 1 - tanh(u)^2
 *
 * which gives:
 *
 * w+ = (X * tanh(X' * w) - sum(sech(X' * w) .^ 2) * w) / X->cols
 * w* = w+ / ||w+||
 *
 * @param w0
 * @param X
 * @return w*
 */
matrix_t * fpica_tanh (matrix_t *w0, matrix_t *X)
{
    // compute w+
    matrix_t *w_temp1 = m_product("w_temp1", X, w0, true, false);
    matrix_t *w_temp2 = m_copy("w_temp2", w_temp1);

    m_elem_apply(w_temp1, tanhf);
    m_elem_apply(w_temp2, sechf);
    m_elem_apply(w_temp2, pow2);

    matrix_t *w_temp3 = m_copy("w_temp3", w0);
    m_elem_mult(w_temp3, m_sum(w_temp2));

    matrix_t *w = m_product("w", X, w_temp1);
    m_subtract(w, w_temp3);
    m_elem_mult(w, 1.0f / X->cols);

    // compute w*
    m_elem_mult(w, 1 / m_norm(w));

    // cleanup
    m_free(w_temp1);
    m_free(w_temp2);
    m_free(w_temp3);

    return w;
}

// TODO: fpica_gauss
// TODO: fpica_skew
// TODO: fpica_relu

/**
 * Compute the mixing matrix W for an input matrix X using
 * the deflation approach. The input matrix should already
 * be whitened.
 *
 * The fixed-point algorithm is defined as follows:
 *
 * w+ = E{X * g(X' * w)} - E{g'(X' * w)} * w
 * w* = w+ / ||w+||
 *
 * @param params
 * @param X
 * @param W_z
 * @return mixing matrix W
 */
matrix_t * fpica (ica_params_t *params, matrix_t *X, matrix_t *W_z)
{
    int vectorSize = X->rows;
    int numSamples = X->cols;

    // if n2 is -1, use default value
    int n2 = (params->n2 == -1)
        ? vectorSize
        : params->n2;

    // determine nonlinearity function
    ica_nonl_func_t fpica_update = NULL;

    if ( params->nonl == ICA_NONL_POW3 ) {
        fpica_update = fpica_pow3;
    }
    else if ( params->nonl == ICA_NONL_TANH ) {
        fpica_update = fpica_tanh;
    }

    matrix_t *B = m_zeros("B", vectorSize, vectorSize);
    matrix_t *W = m_zeros("W", n2, W_z->cols);

    int i;
    for ( i = 0; i < n2; i++ ) {
        log(LL_VERBOSE, "      round %d\n", i + 1);

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
        for ( j = 0; j < params->max_iterations; j++ ) {
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
            if ( norm1 < params->epsilon || norm2 < params->epsilon ) {
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
            m_free(w);

            // compute parameter update
            w = fpica_update(w0, X);
        }

        log(LL_VERBOSE, "      iterations: %d\n", j);

        m_free(w);
        m_free(w0);
    }

    m_free(B);

    return W;
}
