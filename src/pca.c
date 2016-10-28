/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "matrix.h"

/**
 * Compute the principal components of a matrix of image vectors.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix. This function returns all of the n
 * computed eigenvectors, where n is the number of columns.
 *
 * @param X       matrix of mean-subtracted images in columns
 * @param L_eval  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA(matrix_t *X, matrix_t **L_eval)
{
	// compute the surrogate matrix L = X' * X
	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);

	m_free(X_tr);

	// compute eigenvalues, eigenvectors of L
	matrix_t *L_evec;

	m_eigen(L, L_eval, &L_evec);

	// compute principal components W_pca = X * L_evec
	matrix_t *W_pca = m_product(X, L_evec);

	m_free(L);
	m_free(L_evec);

	return W_pca;
}
