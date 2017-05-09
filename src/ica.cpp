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

typedef matrix_t * (*ica_nonl_func_t)(matrix_t *, matrix_t *);

/**
 * Construct an ICA layer.
 *
 * @param n1
 * @param n2
 * @param nonl
 * @param max_iter
 * @param eps
 */
ICALayer::ICALayer(int n1, int n2, ica_nonl_t nonl, int max_iter, precision_t eps)
{
    this->n1 = n1;
    this->n2 = n2;
    this->nonl = nonl;
    this->max_iter = max_iter;
    this->eps = eps;

    this->W = NULL;
}

/**
 * Destruct an ICA layer.
 */
ICALayer::~ICALayer()
{
    m_free(this->W);
}

/**
 * Compute the independent components of a matrix X, which
 * consists of observations in columns.
 *
 * @param X
 * @param y
 * @param c
 * @return independent components of X in columns
 */
matrix_t * ICALayer::compute(matrix_t *X, const std::vector<data_entry_t>& y, int c)
{
    timer_push("ICA");

    timer_push("subtract mean from input matrix");

    // compute mixedsig = X', subtract mean column
    matrix_t *mixedsig = m_transpose("mixedsig", X);
    matrix_t *mixedmean = m_mean_column("mixedmean", mixedsig);

    m_subtract_columns(mixedsig, mixedmean);

    timer_pop();

    timer_push("compute whitening matrix and whitened input matrix");

    // compute whitening matrix W_z = inv(sqrt(D)) * W_pca'
    PCALayer pca(this->n1);

    matrix_t *W_pca = pca.compute(mixedsig, y, c);

    matrix_t *D_temp1 = m_copy("sqrt(D)", pca.D);
    m_elem_apply(D_temp1, sqrtf);

    matrix_t *D_temp2 = m_inverse("inv(sqrt(D))", D_temp1);
    matrix_t *W_z = m_product("W_z", D_temp2, W_pca, false, true);

    // compute whitened input U = W_z * mixedsig
    matrix_t *U = m_product("U", W_z, mixedsig);

    timer_pop();

    timer_push("compute mixing matrix");

    // compute mixing matrix
    matrix_t *W_mix = this->fpica(U, W_z);

    timer_pop();

    timer_push("compute ICA projection matrix");

    // compute independent components
    // icasig = W_mix * (mixedsig + mixedmean * ones(1, mixedsig->cols))
    matrix_t *ones = m_ones("ones", 1, mixedsig->cols);
    matrix_t *icasig_temp1 = m_product("icasig_temp1", mixedmean, ones);

    m_add(icasig_temp1, mixedsig);

    matrix_t *icasig = m_product("icasig", W_mix, icasig_temp1);

    // compute W_ica = icasig'
    this->W = m_transpose("W_ica", icasig);

    timer_pop();

    timer_pop();

    // cleanup
    m_free(mixedsig);
    m_free(mixedmean);
    m_free(D_temp1);
    m_free(D_temp2);
    m_free(W_z);
    m_free(U);
    m_free(W_mix);
    m_free(ones);
    m_free(icasig_temp1);
    m_free(icasig);

    return this->W;
}

/**
 * Project a matrix X into the feature space of an ICA layer.
 *
 * @param X
 * @return projected matrix
 */
matrix_t * ICALayer::project(matrix_t *X)
{
    return m_product("P", this->W, X, true, false);
}

/**
 * Save an ICA layer to a file.
 */
void ICALayer::save(FILE *file)
{
    m_fwrite(file, this->W);
}

/**
 * Load an ICA layer from a file.
 */
void ICALayer::load(FILE *file)
{
    this->W = m_fread(file);
}

/**
 * Print information about an ICA layer.
 */
void ICALayer::print()
{
    const char *nonl_name = "";

    if ( this->nonl == ICA_NONL_POW3 ) {
        nonl_name = "pow3";
    }
    else if ( this->nonl == ICA_NONL_TANH ) {
        nonl_name = "tanh";
    }
    else if ( this->nonl == ICA_NONL_GAUSS ) {
        nonl_name = "gauss";
    }

    log(LL_VERBOSE, "ICA\n");
    log(LL_VERBOSE, "  %-20s  %10d\n", "n1", this->n1);
    log(LL_VERBOSE, "  %-20s  %10d\n", "n2", this->n2);
    log(LL_VERBOSE, "  %-20s  %10s\n", "nonl", nonl_name);
    log(LL_VERBOSE, "  %-20s  %10d\n", "max_iter", this->max_iter);
    log(LL_VERBOSE, "  %-20s  %10f\n", "eps", this->eps);
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
 * w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X->cols
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

/**
 * Gaussian distribution function.
 *
 * @param x
 */
precision_t gauss (precision_t x)
{
    return x * expf(-(x * x) / 2.0f);
}

/**
 * Derivative of the aussian distribution function.
 *
 * @param x
 */
precision_t dgauss (precision_t x)
{
    return (1 - x * x) * expf(-(x * x) / 2.0f);
}

/**
 * Compute the parameter update for fpica
 * with the Gaussian nonlinearity:
 *
 * g(u) = u * exp(-u^2 / 2)
 * g'(u) = (1 - u^2) * exp(-u^2 / 2)
 *
 * which gives:
 *
 * w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X->cols
 * w* = w+ / ||w+||
 *
 * @param w0
 * @param X
 * @return w*
 */
matrix_t * fpica_gauss (matrix_t *w0, matrix_t *X)
{
    // compute w+
    matrix_t *w_temp1 = m_product("w_temp1", X, w0, true, false);
    matrix_t *w_temp2 = m_copy("w_temp2", w_temp1);

    m_elem_apply(w_temp1, gauss);
    m_elem_apply(w_temp2, dgauss);

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

    return w;
}

/**
 * Compute the mixing matrix W_mix for an input matrix X using
 * the deflation approach. The input matrix should already
 * be whitened.
 *
 * @param X
 * @param W_z
 * @return mixing matrix W_mix
 */
matrix_t * ICALayer::fpica(matrix_t *X, matrix_t *W_z)
{
    // if n2 is -1, use default value
    int n2 = (this->n2 == -1)
        ? X->rows
        : min(X->rows, this->n2);

    // determine nonlinearity function
    ica_nonl_func_t fpica_update = NULL;

    if ( this->nonl == ICA_NONL_POW3 ) {
        fpica_update = fpica_pow3;
    }
    else if ( this->nonl == ICA_NONL_TANH ) {
        fpica_update = fpica_tanh;
    }
    else if ( this->nonl == ICA_NONL_GAUSS ) {
        fpica_update = fpica_gauss;
    }

    matrix_t *B = m_zeros("B", n2, n2);
    matrix_t *W_mix = m_zeros("W_mix", n2, W_z->cols);

    int i;
    for ( i = 0; i < n2; i++ ) {
        log(LL_VERBOSE, "      round %d\n", i + 1);

        // initialize w as a Gaussian (0, 1) random vector
        matrix_t *w = m_random("w", n2, 1);

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
        for ( j = 0; j < this->max_iter; j++ ) {
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
            if ( norm1 < this->eps || norm2 < this->eps ) {
                // save B(:, i) = w
                m_assign_column(B, i, w, 0);

                // save W_mix(i, :) = w' * W_z
                matrix_t *W_temp1 = m_product("W_temp1", w, W_z, true, false);

                m_assign_row(W_mix, i, W_temp1, 0);

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

    return W_mix;
}
