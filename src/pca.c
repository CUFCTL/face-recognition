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
 * @param X  mean-subtracted image matrix
 * @return projection matrix W_pca'
 */
matrix_t * PCA(matrix_t *X)
{
	// compute the surrogate matrix L = X' * X
	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);

	m_free(X_tr);

	// compute eigenvectors for L
	matrix_t *L_eval = m_initialize(L->rows, 1);
	matrix_t *L_evec = m_initialize(L->rows, L->cols);

	m_eigen(L, L_eval, L_evec);

	// compute eigenfaces W_pca = X * L_evec
	matrix_t *W_pca = m_product(X, L_evec);
	matrix_t *W_pca_tr = m_transpose(W_pca);

	m_free(L);
	m_free(L_eval);
	m_free(L_evec);
	m_free(W_pca);

	return W_pca_tr;
}
