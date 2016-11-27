/**
 * @file lda.c
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include "database.h"
#include "timer.h"
#include <stdlib.h>

/**
 * Compute the scatter matrices S_b and S_w for a matrix X.
 *
 * @param X        input matrix
 * @param c        number of classes
 * @param entries  list of entries for each column of X
 * @param p_S_b    pointer to store between-scatter matrix
 * @param p_S_w    pointer to store within-scatter matrix
 */
void m_scatter(matrix_t *X, int c, image_entry_t *entries, matrix_t **p_S_b, matrix_t **p_S_w)
{
    matrix_t **X_classes = (matrix_t **)malloc(c * sizeof(matrix_t *));
    matrix_t **U = (matrix_t **)malloc(c * sizeof(matrix_t *));

    // compute the mean of each class
    int i, j;
    for ( i = 0, j = 0; i < c; i++ ) {
        // extract the columns of X in class i
        int k;
        for ( k = j; k < X->cols; k++ ) {
            if ( entries[k].label->id != entries[j].label->id ) {
                break;
            }
        }

        X_classes[i] = m_copy_columns(X, j, k);
        j = k;

        // compute the mean of the class
        U[i] = m_mean_column(X_classes[i]);
    }

    // compute the mean of all classes
    matrix_t *u = m_initialize(X->rows, 1);  // m_mean_column(U);

    for ( i = 0; i < c; i++ ) {
        m_add(u, U[i]);
    }
    m_elem_mult(u, 1.0f / c);

    // compute the between-scatter S_b = sum(S_b_i, i=1:c)
    // compute the within-scatter S_w = sum(S_w_i, i=1:c)
    matrix_t *S_b = m_zeros(X->rows, X->rows);
    matrix_t *S_w = m_zeros(X->rows, X->rows);

    for ( i = 0; i < c; i++ ) {
        matrix_t *X_class = X_classes[i];

        // compute S_b_i = n_i * (u_i - u) * (u_i - u)'
        matrix_t *u_i = m_copy(U[i]);
        m_subtract(u_i, u);

        matrix_t *S_b_i = m_product(u_i, u_i, false, true);
        m_elem_mult(S_b_i, X_class->cols);

        m_add(S_b, S_b_i);

        // compute S_w_i = X_class * X_class', X_class is mean-subtracted
        m_subtract_columns(X_class, U[i]);

        matrix_t *S_w_i = m_product(X_class, X_class, false, true);

        m_add(S_w, S_w_i);

        // cleanup
        m_free(u_i);
        m_free(S_b_i);
        m_free(S_w_i);
    }

    // save outputs
    *p_S_b = S_b;
    *p_S_w = S_w;

    // cleanup
    for ( i = 0; i < c; i++ ) {
        m_free(X_classes[i]);
        m_free(U[i]);
    }
    free(X_classes);
    free(U);
    m_free(u);
}

/**
 * Compute the projection matrix of a training set with LDA.
 *
 * @param W_pca    PCA projection matrix
 * @param X        image matrix
 * @param c        number of classes
 * @param entries  list of entries for each image
 * @param n_opt1   number of columns in W_pca to use
 * @param n_opt2   number of columns in W_fld to use
 * @return projection matrix W_lda
 */
matrix_t * LDA(matrix_t *W_pca, matrix_t *X, int c, image_entry_t *entries, int n_opt1, int n_opt2)
{
    timer_push("  LDA");

    timer_push("    truncate eigenfaces and projected images");

    // if n_opt1 = -1, use N - c
    n_opt1 = (n_opt1 == -1)
        ? X->cols - c
        : n_opt1;

    // use only the last n_opt1 columns in W_pca
    matrix_t *W_pca2 = m_copy_columns(W_pca, W_pca->cols - n_opt1, W_pca->cols);
    matrix_t *P_pca = m_product(W_pca2, X, true, false);

    timer_pop();

    timer_push("    compute scatter matrices");

    // compute scatter matrices S_b and S_w
    matrix_t *S_b;
    matrix_t *S_w;

    m_scatter(P_pca, c, entries, &S_b, &S_w);

    timer_pop();

    timer_push("    compute eigenvectors of scatter matrices");

    // compute W_fld = eigenvectors of S_w^-1 * S_b
    matrix_t *S_w_inv = m_inverse(S_w);
    matrix_t *J = m_product(S_w_inv, S_b);
    matrix_t *J_evec;
    matrix_t *J_eval;

    m_eigen(J, &J_evec, &J_eval);

    // TODO: compare results, see if this will work
    // m_eigen2(S_b, S_w, &J_evec, &J_eval);

    // if n_opt2 = -1, use c - 1
    n_opt2 = (n_opt2 == -1)
        ? c - 1
        : n_opt2;

    // take only the last n_opt2 columns in J_evec
    matrix_t *W_fld = m_copy_columns(J_evec, J_evec->cols - n_opt2, J_evec->cols);

    timer_pop();

    timer_push("    compute LDA projection matrix");

    // compute W_lda = (W_fld' * W_pca2')' = W_pca2 * W_fld
    matrix_t *W_lda = m_product(W_pca2, W_fld);

    timer_pop();

    timer_pop();

    // cleanup
    m_free(W_pca2);
    m_free(P_pca);
    m_free(S_b);
    m_free(S_w);
    m_free(S_w_inv);
    m_free(J);
    m_free(J_eval);
    m_free(J_evec);
    m_free(W_fld);

    return W_lda;
}
