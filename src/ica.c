/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>

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

/**
 * Compute the 2-norm of a matrix.
 *
 * @param M  pointer to matrix
 * @return 2-norm of M
 */
precision_t m_norm(matrix_t *M)
{
    precision_t norm = 0;

    int i, j;
    for ( i = 0; i < M->rows; i++ ) {
        for ( j = 0; j < M->cols; j++ ) {
            norm += elem(M, i, j) * elem(M, i, j);
        }
    }

    return sqrt(norm);
}

/**
 * Compute the "angle" between two matrices.
 *
 * A and B are treated as column vectors, so that
 * the angle between them is:
 *
 * acos(-A * B / (||A|| * ||B||))
 *
 * @param A  pointer to matrix
 * @param B  pointer to matrix
 * @return angle between A and B
 */
precision_t m_angle(matrix_t *A, matrix_t *B)
{
    assert(A->rows == B->rows && A->cols == B->cols);

    // compute A * B
    precision_t a_dot_b = 0;

    int i, j;
    for ( i = 0; i < A->rows; i++ ) {
        for ( j = 0; j < A->cols; j++ ) {
            a_dot_b += elem(A, i, j) * elem(B, i, j);
        }
    }

    return acos(-a_dot_b / (m_norm(A) * m_norm(B)));
}

/**
 * Implementation of the learning rule described in Bell & Sejnowski,
 * Vision Research, in press for 1997, that contained the natural
 * gradient (W' * W).
 *
 * Bell & Sejnowski hold the patent for this learning rule.
 *
 * SEP goes once through the mixed signals X in batch blocks of size B,
 * adjusting weights W at the end of each block.
 *
 * sepout is called every F counts.
 *
 * I suggest a learning rate (L) of 0.006, and a block size (B) of
 * 300, at least for 2->2 separation.  When annealing to the right
 * solution for 10->10, however, L < 0.0001 and B = 10 were most successful.
 *
 * @param X  "sphered" input matrix
 * @param W  weight matrix
 * @param B  block size
 * @param L  learning rate
 * @param F  interval to print training stats
 */
void sep96(matrix_t *X, matrix_t *W, int B, double L, int F)
{
    matrix_t *BI = m_identity(X->rows);
    m_elem_mult(BI, B);

    int t;
    for ( t = 0; t < X->cols; t += B ) {
        int end = (t + B < X->cols)
            ? t + B
            : X->cols;

        matrix_t *W0 = m_copy(W);
        matrix_t *X_batch = m_copy_columns(X, t, end);
        matrix_t *U = m_product(W0, X_batch);

        // compute Y' = 1 - 2 * f(U), f(u) = 1 / (1 + e^(-u))
        matrix_t *Y_p = m_initialize(U->rows, U->cols);

        int i, j;
        for ( i = 0; i < Y_p->rows; i++ ) {
            for ( j = 0; j < Y_p->cols; j++ ) {
                elem(Y_p, i, j) = 1 - 2 * (1 / (1 + exp(-elem(U, i, j))));
            }
        }

        // compute dW = L * (BI + Y'U') * W0
        matrix_t *U_tr = m_transpose(U);
        matrix_t *dW_temp1 = m_product(Y_p, U_tr);
        m_add(dW_temp1, BI);

        matrix_t *dW = m_product(dW_temp1, W0);
        m_elem_mult(dW, L);

        // compute W = W0 + dW
        m_add(W, dW);

        // print training stats
        if ( t % F == 0 ) {
            precision_t norm = m_norm(dW);
            precision_t angle = m_angle(W0, W);

            printf("*** norm(dW) = %.4lf, angle(W0, W) = %.1lf deg\n", norm, 180 * angle / M_PI);
        }

        // cleanup
        m_free(W0);
        m_free(X_batch);
        m_free(U);
        m_free(Y_p);
        m_free(U_tr);
        m_free(dW_temp1);
        m_free(dW);
    }
}

typedef struct sep96_params {
    int B;
    double L;
    int F;
    int N;
} sep96_params_t;

/**
 * Compute the ICA weight matrix W_I for an input matrix X.
 *
 * @param X  mean-subtracted input matrix
 * @return weight matrix W_I
 */
matrix_t * run_ica(matrix_t *X)
{
    // compute whitening matrix W_z
    matrix_t *W_z = sphere(X);
    matrix_t *X_sph = m_product(W_z, X);

    // shuffle the columns of X_sph
    m_shuffle_columns(X_sph);

    // train the weight matrix W
    matrix_t *W = m_identity(X->rows);

    sep96_params_t params[] = {
        { 50, 0.0005, 5000, 1000 },
        { 50, 0.0003, 5000, 200 },
        { 50, 0.0002, 5000, 200 },
        { 50, 0.0001, 5000, 200 }
    };
    int num_sweeps = sizeof(params) / sizeof(sep96_params_t);

    int i, j;
    for ( i = 0; i < num_sweeps; i++ ) {
        printf("sweep %d: B = %d, L = %lf\n", i + 1, params[i].B, params[i].L);

        for ( j = 0; j < params[i].N; j++ ) {
            sep96(X_sph, W, params[i].B, params[i].L, params[i].F);
        }
    }

    // compute W_I = W * W_z
    matrix_t *W_I = m_product(W, W_z);

    // cleanup
    m_free(W_z);
    m_free(X_sph);
    m_free(W);

    return W_I;
}

/**
 * Compute the projection matrix of a training set with ICA
 * using Architecture II.
 *
 * @param W_pca_tr  PCA projection matrix
 * @param P_pca     PCA projected images
 * @return projection matrix W_ica'
 */
matrix_t * ICA2(matrix_t *W_pca_tr, matrix_t *P_pca)
{
    // compute weight matrix W_I
    matrix_t *W_I = run_ica(P_pca);

    // compute W_ica' = W_I * W_pca'
    matrix_t *W_ica_tr = m_product(W_I, W_pca_tr);

    // cleanup
    m_free(W_I);

    return W_ica_tr;
}
