/**
 * @file pca.cpp
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "math_helper.h"
#include "pca.h"
#include "timer.h"

/**
 * Compute the principal components of a matrix X, which
 * consists of observations in rows or columns. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param params
 * @param X
 * @param p_D
 * @return principal components of X in columns
 */
matrix_t * PCA(pca_params_t *params, matrix_t *X, matrix_t **p_D)
{
	matrix_t *W_pca;
	matrix_t *D;

	// if n1 = -1, use default value
	int n1 = (params->n1 == -1)
		? min(X->rows, X->cols)
		: params->n1;

	timer_push("PCA");

	if ( X->rows > X->cols ) {
		timer_push("compute surrogate of covariance matrix L");

		matrix_t *L = m_product("L", X, X, true, false);

		timer_pop();

		timer_push("compute eigenvectors of L");

		matrix_t *V;
		m_eigen("V", "D", L, n1, &V, &D);

		timer_pop();

		timer_push("compute principal components");

		W_pca = m_product("W_pca", X, V);

		timer_pop();

		// cleanup
		m_free(L);
		m_free(V);
	}
	else {
		timer_push("compute covariance matrix C");

		matrix_t *C = m_product("C", X, X, false, true);

		timer_pop();

		timer_push("compute eigendecomposition of C");

		m_eigen("W_pca", "D", C, n1, &W_pca, &D);

		timer_pop();

		// cleanup
		m_free(C);
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
