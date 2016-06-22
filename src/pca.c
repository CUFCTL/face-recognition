/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "matrix.h"

/**
 * Compute the principal components of a training set.
 *
 * Currently, this function returns all of the n computed
 * eigenvectors, where n is the number of training images.
 *
 * @param A  mean-subtracted image matrix
 * @return projection matrix W_pca
 */
matrix_t * get_projection_matrix_PCA(matrix_t *A)
{
	// compute the surrogate matrix L = A' * A
	matrix_t *A_tr = m_transpose(A);
	matrix_t *L = m_product(A_tr, A);

	m_free(A_tr);

	// compute eigenvectors for L
	matrix_t *L_eval = m_initialize(L->rows, 1);
	matrix_t *L_evec = m_initialize(L->rows, L->cols);

	m_eigenvalues_eigenvectors(L, L_eval, L_evec);

	// compute eigenfaces W_pca = A * L_evec
	matrix_t *W_pca = m_product(A, L_evec);

	m_free(L);
	m_free(L_eval);
	m_free(L_evec);

	return W_pca;
}