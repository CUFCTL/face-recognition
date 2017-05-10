/**
 * @file matrix_utils.cpp
 *
 * Library of helpful matrix functions.
 */
#include <assert.h>
#include <stdlib.h>
#include "matrix_utils.h"

/**
 * Copy a matrix X into a list X_c of class
 * submatrices.
 *
 * This function assumes that the columns in X
 * are grouped by class.
 *
 * @param X
 * @param y
 * @param c
 * @return list of submatrices
 */
matrix_t ** m_copy_classes(matrix_t *X, const std::vector<data_entry_t>& y, int c)
{
	matrix_t **X_c = (matrix_t **)malloc(c * sizeof(matrix_t *));

	int i, j;
	for ( i = 0, j = 0; i < c; i++ ) {
		int k = j;
		while ( k < X->cols && y[k].label == y[j].label ) {
			k++;
		}

		X_c[i] = m_copy_columns("X_c_i", X, j, k);
		j = k;
	}

	assert(j == X->cols);

	return X_c;
}

/**
 * Compute the mean of each class for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * @param X_c
 * @param c
 * @return list of column vectors
 */
matrix_t ** m_class_means(matrix_t **X_c, int c)
{
	matrix_t **U = (matrix_t **)malloc(c * sizeof(matrix_t *));

	int i;
	for ( i = 0; i < c; i++ ) {
		U[i] = m_mean_column("U_i", X_c[i]);
	}

	return U;
}

/**
 * Compute the class covariance matrices for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_i = (X_c_i - U_i) * (X_c_i - U_i)'
 *
 * @param X_c
 * @param U
 * @param c
 * @return list of class covariance matrices
 */
matrix_t ** m_class_scatters(matrix_t **X_c, matrix_t **U, int c)
{
	matrix_t **S = (matrix_t **) malloc(c * sizeof(matrix_t *));

	int i;
	for ( i = 0; i < c; i++ ) {
		matrix_t *X_c_i = m_copy("X_c_i", X_c[i]);
		m_subtract_columns(X_c_i, U[i]);

		S[i] = m_product("S_i", X_c_i, X_c_i, false, true);

		m_free(X_c_i);
	}

	return S;
}

/**
 * Compute the between-scatter matrix S_b for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_b = sum(n_i * (u_i - u) * (u_i - u)', i=1:c),
 *
 * @param X_c
 * @param U
 * @param c
 * @return between-scatter matrix
 */
matrix_t * m_scatter_between(matrix_t **X_c, matrix_t **U, int c)
{
	int N = U[0]->rows;

	// compute the mean of all classes
	matrix_t *u = m_initialize("u", N, 1);

	int i;
	for ( i = 0; i < c; i++ ) {
		m_add(u, U[i]);
	}
	m_elem_mult(u, 1.0f / c);

	// compute the between-scatter S_b
	matrix_t *S_b = m_zeros("S_b", N, N);

	for ( i = 0; i < c; i++ ) {
		// compute S_b_i
		matrix_t *u_i = m_copy("u_i - u", U[i]);
		m_subtract(u_i, u);

		matrix_t *S_b_i = m_product("S_b_i", u_i, u_i, false, true);
		m_elem_mult(S_b_i, X_c[i]->cols);

		m_add(S_b, S_b_i);

		// cleanup
		m_free(u_i);
		m_free(S_b_i);
	}

	// cleanup
	m_free(u);

	return S_b;
}

/**
 * Compute the within-scatter matrix S_w for a matrix X,
 * given by a list X_c of class submatrices.
 *
 * S_w = sum((X_c_i - U_i) * (X_c_i - U_i)', i=1:c)
 *
 * @param X_c
 * @param U
 * @param c
 * @return within-scatter matrix
 */
matrix_t * m_scatter_within(matrix_t **X_c, matrix_t **U, int c)
{
	// compute the within-scatter S_w
	int N = U[0]->rows;
	matrix_t *S_w = m_zeros("S_w", N, N);

	int i;
	for ( i = 0; i < c; i++ ) {
		// compute S_w_i
		matrix_t *X_c_i = m_copy("X_c_i", X_c[i]);
		m_subtract_columns(X_c_i, U[i]);

		matrix_t *S_w_i = m_product("S_w_i", X_c_i, X_c_i, false, true);

		m_add(S_w, S_w_i);

		// cleanup
		m_free(X_c_i);
		m_free(S_w_i);
	}

	return S_w;
}
