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
 * @param X    matrix of mean-subtracted images in columns
 * @param p_D  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA(matrix_t *X, matrix_t **p_D)
{
	timing_start("PCA");

	timing_start("Find surrogate matrix L");

	// compute the surrogate matrix L = X' * X
	if ( VERBOSE ) {
		printf("\tFinding Surrogate Matrix...\n");
	}

	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);

	m_free(X_tr);

	timing_end("Find surrogate matrix L");

	timing_start("Compute Eigenspace of training set");

	// compute eigenvalues, eigenvectors of L
	if ( VERBOSE ) {
		printf("\tComputing Eigenspace of training set...\n");
	}

	matrix_t *V;
	matrix_t *D;

	m_eigen(L, &V, &D);

	timing_end("Compute Eigenspace of training set");

	timing_start("Project Training Set onto Eigenspace");

	// compute principal components W_pca = X * V
	if ( VERBOSE ) {
		printf("\tProjecting training set onto Eigenspace...\n");
	}

	matrix_t *W_pca = m_product(X, V);

	timing_end("Project Training Set onto Eigenspace");

	timing_end("PCA");

	// save outputs
	*p_D = D;

	// cleanup
	m_free(L);
	m_free(V);

	return W_pca;
}
