/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "timer.h"

/**
 * Compute the principal components of a matrix X, which
 * consists of observations in columns. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param X       input matrix
 * @param n_opt1  number of columns in W_pca to use
 * @param p_D     pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA_cols(matrix_t *X, int n_opt1, matrix_t **p_D)
{
	timer_push("  PCA");

	timer_push("    compute surrogate matrix L");

	matrix_t *L = m_product(X, X, true, false);

	timer_pop();

	timer_push("    compute eigenvectors of L");

	matrix_t *V;
	matrix_t *D;

	m_eigen(L, &V, &D);

	timer_pop();

	timer_push("    compute PCA projection matrix");

	// if n_opt1 = -1, use N - 1
	n_opt1 = (n_opt1 == -1)
		? X->cols - 1
		: n_opt1;

	matrix_t *W_pca = m_product(X, V);
	matrix_t *W_pca2 = m_copy_columns(W_pca, W_pca->cols - n_opt1, W_pca->cols);

	timer_pop();

	timer_pop();

	// save outputs
	*p_D = D;

	// cleanup
	m_free(L);
	m_free(V);
	m_free(W_pca);

	return W_pca2;
}

/**
 * Compute the principal components of a matrix X, which
 * consists of observations in rows. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param X    input matrix
 * @param p_D  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA_rows(matrix_t *X, matrix_t **p_D)
{
    // compute the covariance of X
    matrix_t *C = m_covariance(X);

    // compute the eigenvalues, eigenvectors of the covariance
    matrix_t *E;
    matrix_t *D;

    m_eigen(C, &E, &D);

    // save outputs
    *p_D = D;

    // cleanup
    m_free(C);

    return E;
}
