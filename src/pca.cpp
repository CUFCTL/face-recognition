/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "timing.h"

/**
 * Compute the principal components of a matrix of image vectors.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix. This function returns all of the n
 * computed eigenvectors, where n is the number of columns.
 *
 * @param X    matrix of mean-subtracted images in columns
 * @param p_D  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA(matrix_t *X, matrix_t **p_D)
{
	timing_push("  PCA");

	timing_push("    compute surrogate matrix L");

	// compute the surrogate matrix L = X' * X
	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);

	m_free(X_tr);

	timing_pop();

	timing_push("    compute eigenvectors of L");

	// compute eigenvalues, eigenvectors of L
	matrix_t *V;
	matrix_t *D;

	m_eigen(L, &V, &D);

	timing_pop();

	timing_push("    compute PCA projection matrix");

	// compute principal components W_pca = X * V
	matrix_t *W_pca = m_product(X, V);

	timing_pop();

	timing_pop();

	// save outputs
	*p_D = D;

	// cleanup
	m_free(L);
	m_free(V);

	return W_pca;
}
