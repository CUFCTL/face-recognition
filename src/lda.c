/**
 * @file lda.c
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include "database.h"
#include "matrix.h"
#include <stdlib.h>

/**
 * Compute the scatter matrices S_w and S_b for a set of images.
 *
 * @param db
 * @param S_b
 * @param S_w
 */
void m_scatter(database_t *db, matrix_t *S_b, matrix_t *S_w)
{
    matrix_t *M = db->P_pca;
    matrix_t **M_classes = (matrix_t **)malloc(db->num_classes * sizeof(matrix_t *));
    matrix_t **U = (matrix_t **)malloc(db->num_classes * sizeof(matrix_t *));

    // compute the mean of each class
    int i, j;
    for ( i = 0, j = 0; i < db->num_classes; i++ ) {
        // extract the columns of M in class i
        int k;
        for ( k = j; k < M->cols; k++ ) {
            if ( db->entries[k].class != db->entries[j].class ) {
                break;
            }
        }

        M_classes[i] = m_copy_columns(M, j, k);
        j = k;

        // compute the mean of the class
        U[i] = m_mean_column(M_classes[i]);
    }

    // compute the mean of all classes
    matrix_t *u = m_initialize(M->rows, 1);  // m_mean_column(U);

	for ( i = 0; i < db->num_classes; i++ ) {
		for ( j = 0; j < M->rows; j++ ) {
			elem(u, j, 0) += elem(U[i], j, 0);
		}
	}
	for ( i = 0; i < M->rows; i++ ) {
		elem(u, i, 0) /= db->num_classes;
	}

    // compute the scatter matrices S_b and S_w
    for ( i = 0; i < db->num_classes; i++ ) {
        matrix_t *M_class = M_classes[i];

        // compute S_b_i = n_i * (u_i - u) * (u_i - u)'
        matrix_t *u_i_norm = m_copy(U[i]);
        m_subtract(u_i_norm, u);

        matrix_t *u_i_norm_tr = m_transpose(u_i_norm);
        matrix_t *S_b_i = m_product(u_i_norm, u_i_norm_tr);
        m_elem_mult(S_b_i, M_class->cols);

        m_add(S_b, S_b_i);

        // compute S_w_i = M_class_norm * M_class_norm'
        m_subtract_columns(M_class, U[i]);

        matrix_t *M_class_tr = m_transpose(M_class);
        matrix_t *S_w_i = m_product(M_class, M_class_tr);

        m_add(S_w, S_w_i);

        // cleanup
        m_free(u_i_norm);
        m_free(u_i_norm_tr);
        m_free(S_b_i);
        m_free(M_class_tr);
        m_free(S_w_i);
    }

    // cleanup
    for ( i = 0; i < db->num_classes; i++ ) {
        m_free(M_classes[i]);
        m_free(U[i]);
    }
    free(M_classes);
    free(U);
}

/**
 * Compute the projection matrix of a training set with LDA.
 *
 * @param db  pointer to database
 * @return projection matrix W_opt'
 */
matrix_t * LDA(database_t *db)
{
    // TODO: take only the first n - c columns of P_pca

    // compute scatter matrices S_b and S_w
    matrix_t *S_b = m_zeros(db->P_pca->rows, db->P_pca->rows);
    matrix_t *S_w = m_zeros(db->P_pca->rows, db->P_pca->rows);

    m_scatter(db, S_b, S_w);

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
    matrix_t *W_opt_tr = m_product(W_lda_tr, db->W_pca_tr);

    m_free(S_b);
    m_free(S_w);
    m_free(S_w_inv);
    m_free(J);
    m_free(J_eval);
    m_free(J_evec); // W_lda
    m_free(W_lda_tr);

    return W_opt_tr;
}
