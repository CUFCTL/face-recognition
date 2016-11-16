/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "matrix.h"
#include "timing.h"

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
	timing_start("PCA");

	if ( VERBOSE ) {
		printf("\tFinding Surrogate Matrix of L...\n");
	}
	// compute the surrogate matrix L = X' * X
	timing_start("Find surrogate of L");
	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);
	timing_end("Find surrogate of L");

	m_free(X_tr);

	// compute eigenvalues, eigenvectors of L
	matrix_t *L_evec;

	if ( VERBOSE ) {
		printf("\tComputing Eigenspace of training set...\n");
	}
	timing_start("Compute Eigenspace of training set");
	m_eigen(L, L_eval, &L_evec);
	timing_end("Compute Eigenspace of training set");

	if ( VERBOSE ) {
		printf("\tProjecting training set onto Eigenspace...\n");
	}
	// compute principal components W_pca = X * L_evec
	timing_start("Project Training Set onto Eigenspace");
	matrix_t *W_pca = m_product(X, L_evec);
	timing_end("Project Training Set onto Eigenspace");

	timing_end("PCA");

	m_free(L);
	m_free(L_evec);

	return W_pca;
}
