/**
 * @file lda.c
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include "database.h"
#include "matrix.h"

void m_scatter(matrix_t *M, matrix_t *S_b, matrix_t *S_w)
{
    int num_classes;

    // compute the mean within each class
    matrix_t *U;

    int i;
    for ( i = 0; i < num_classes; i++ ) {
        // extract the columns of M in class i
        matrix_t *M_class;

        // U[i] = m_mean_column(M_class)
    }

    // compute the mean of all classes
    matrix_t *u = m_mean_column(U);

    for ( i = 0; i < num_classes; i++ ) {
        // compute S_b_i = n_i * (u_i - u) * (u_i - u)'
        int n_i;
        matrix_t *u_i;
        matrix_t *u_i_norm = m_copy(u_i);
        m_normalize_columns(u_i_norm, u);

        matrix_t *u_i_norm_tr = m_transpose(u_i_norm);
        matrix_t *S_b_i = m_product(u_i_norm, u_i_norm_tr);
        m_elem_mult(S_b_i, n_i);

        m_add(S_b, S_b_i);

        // compute S_w_i = M_class_norm * M_class_norm'
        // extract the columns of M in class i
        matrix_t *M_class;
        m_normalize_columns(M_class, u_i);

        matrix_t *M_class_tr = m_transpose(M_class);
        matrix_t *S_w_i = m_product(M_class, M_class_tr);

        m_add(S_w, S_w_i);
    }
}

/**
 * Compute the projection matrix of a training set with LDA.
 *
 * @param W_pca_tr  transposed projection matrix from PCA
 * @param P_pca     projected images matrix from PCA
 * @param c         number of classes
 * @return projection matrix W_opt'
 */
matrix_t * get_projection_matrix_LDA(matrix_t *W_pca_tr, matrix_t *P_pca, int c)
{
    // TODO: take only the first n - c columns of P_pca

    // compute scatter matrices S_b and S_w
    matrix_t *S_b = m_zeros(P_pca->rows, P_pca->rows);
    matrix_t *S_w = m_zeros(P_pca->rows, P_pca->rows);

    m_scatter(P_pca, S_b, S_w);

    // compute W_lda = eigenvectors of S_w^-1 * S_b
    matrix_t *S_w_inv = m_inverse(S_w);
    matrix_t *J = m_product(S_w_inv, S_b);
    matrix_t *J_eval = m_initialize(J->rows, 1);
    matrix_t *J_evec = m_initialize(J->rows, J->cols);

    m_eigenvalues_eigenvectors(J, J_eval, J_evec);

    // TODO: take only the first c - 1 columns of J_evec
    matrix_t *W_lda = J_evec;

    matrix_t *W_lda_tr = m_transpose(W_lda);

    // compute W_opt' = W_lda' * W_pca'
    matrix_t *W_opt_tr = m_product(W_lda_tr, W_pca_tr);

    m_free(S_b);
    m_free(S_w);
    m_free(S_w_inv);
    m_free(J);
    m_free(J_eval);
    m_free(J_evec); // W_lda
    m_free(W_lda_tr);

    return W_opt_tr;
}
