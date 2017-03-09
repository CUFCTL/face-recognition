/**
 * @file pca.cpp
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
 * @param X    input matrix
 * @param n1   number of columns to use in W_pca
 * @param p_D  pointer to save eigenvalues
 * @return principal components of X in columns
 */
matrix_t * PCA(matrix_t *X, int n1, matrix_t **p_D)
{
	matrix_t *W_pca;
	matrix_t *D;

	timer_push("  PCA");

	if ( X->rows > X->cols ) {
		timer_push("    compute surrogate of covariance matrix L");

		matrix_t *L = m_product("L", X, X, true, false);

		timer_pop();

		timer_push("    compute eigenvectors of L");

		matrix_t *V;
		m_eigen("V", "D", L, &V, &D);

		timer_pop();

		timer_push("    compute PCA projection matrix");

		// if n1 = -1, use default value
		n1 = (n1 == -1)
			? D->cols
			: n1;

		matrix_t *W_pca_temp1 = m_product("W_pca", X, V);
		W_pca = m_copy_columns("W_pca", W_pca_temp1, W_pca_temp1->cols - n1, W_pca_temp1->cols);

		timer_pop();

		// cleanup
		m_free(L);
		m_free(V);
		m_free(W_pca_temp1);
	}
	else {
		timer_push("    compute covariance matrix C");

		matrix_t *C = m_product("C", X, X, false, true);

		timer_pop();

		timer_push("    compute eigendecomposition of C");

		matrix_t *V;
		m_eigen("V", "D", C, &V, &D);

		timer_pop();

		timer_push("    compute PCA projection matrix");

		// if n1 = -1, use default value
		n1 = (n1 == -1)
			? D->cols
			: n1;

		W_pca = m_copy_columns("W_pca", V, V->cols - n1, V->cols);

		timer_pop();

		// cleanup
		m_free(V);
	}

	timer_pop();

	// save outputs
	if ( p_D != NULL ) {
		*p_D = D;
	}
	else {
		m_free(D);
	}

	return W_pca;
}
