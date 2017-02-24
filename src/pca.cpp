/**
 * @file pca.cpp
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "math_helper.h"
#include "timer.h"

/**
 * Compute the principal components of a matrix X, which
 * consists of observations in columns. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param X    input matrix
 * @param n1   number of columns to use in W_pca
 * @param p_D  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA_cols(matrix_t *X, int n1, matrix_t **p_D)
{
	timer_push("  PCA");

	timer_push("    compute surrogate of covariance matrix L");

	matrix_t *L = m_product("L", X, X, true, false);

	timer_pop();

	timer_push("    compute eigenvectors of L");

	matrix_t *V;
	matrix_t *D;

	m_eigen("V", "D", L, &V, &D);

	timer_pop();

	timer_push("    compute PCA projection matrix");

	// if n1 = -1, use default value
	n1 = (n1 == -1)
		? min(D->cols, X->cols - 1)
		: n1;

	matrix_t *W_pca_temp1 = m_product("W_pca", X, V);
	matrix_t *W_pca = m_copy_columns("W_pca", W_pca_temp1, W_pca_temp1->cols - n1, W_pca_temp1->cols);

	timer_pop();

	timer_pop();

	// save outputs
	*p_D = D;

	// cleanup
	m_free(L);
	m_free(V);
	m_free(W_pca_temp1);

	return W_pca;
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
    matrix_t *C = m_covariance("C", X);

    // compute the eigenvalues, eigenvectors of the covariance
    matrix_t *E;
    matrix_t *D;

    m_eigen("E", "D", C, &E, &D);

    // save outputs
    *p_D = D;

    // cleanup
    m_free(C);

    return E;
}
