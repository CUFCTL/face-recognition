/**
 * @file lda.cpp
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include <assert.h>
#include <stdlib.h>
#include "lda.h"
#include "logger.h"
#include "pca.h"
#include "timer.h"

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

/**
 * Construct an LDA layer.
 *
 * @param n1
 * @param n2
 */
LDALayer::LDALayer(int n1, int n2)
{
	this->n1 = n1;
	this->n2 = n2;

	this->W = NULL;
}

/**
 * Destruct an LDA layer.
 */
LDALayer::~LDALayer()
{
	m_free(this->W);
}

/**
 * Compute the LDA features of a matrix X.
 *
 * @param X
 * @param y
 * @param c
 * @return projection matrix W_lda
 */
matrix_t * LDALayer::compute(matrix_t *X, const std::vector<data_entry_t>& y, int c)
{
	// if n1 = -1, use default value
	int n1 = (this->n1 == -1)
		? X->cols - c
		: this->n1;

	// if n2 = -1, use default value
	int n2 = (this->n2 == -1)
		? c - 1
		: this->n2;

	if ( n1 <= 0 ) {
		log(LL_ERROR, "error: training set is too small for LDA\n");
		exit(1);
	}

	timer_push("LDA");

	timer_push("compute eigenfaces");

	PCALayer pca(n1);
	matrix_t *W_pca = pca.compute(X, y, c);
	matrix_t *P_pca = pca.project(X);

	timer_pop();

	timer_push("compute scatter matrices S_b and S_w");

	matrix_t **X_c = m_copy_classes(P_pca, y, c);
	matrix_t **U = m_class_means(X_c, c);
	matrix_t *S_b = m_scatter_between(X_c, U, c);
	matrix_t *S_w = m_scatter_within(X_c, U, c);

	timer_pop();

	timer_push("compute eigendecomposition of S_b and S_w");

	matrix_t *S_w_inv = m_inverse("inv(S_w)", S_w);
	matrix_t *J = m_product("J", S_w_inv, S_b);

	matrix_t *W_fld;
	matrix_t *J_eval;
	m_eigen("W_fld", "J_eval", J, n2, &W_fld, &J_eval);

	timer_pop();

	timer_push("compute Fisherfaces");

	this->W = m_product("W_lda", W_pca, W_fld);

	timer_pop();

	timer_pop();

	// cleanup
	m_free(P_pca);

	int i;
	for ( i = 0; i < c; i++ ) {
		m_free(X_c[i]);
		m_free(U[i]);
	}
	free(X_c);
	free(U);

	m_free(S_b);
	m_free(S_w);
	m_free(S_w_inv);
	m_free(J);
	m_free(W_fld);
	m_free(J_eval);

	return this->W;
}

/**
 * Project a matrix X into the feature space of an LDA layer.
 *
 * @param X
 * @return projected matrix
 */
matrix_t * LDALayer::project(matrix_t *X)
{
	return m_product("P", this->W, X, true, false);
}

/**
 * Save an LDA layer to a file.
 */
void LDALayer::save(FILE *file)
{
	m_fwrite(file, this->W);
}

/**
 * Load an LDA layer from a file.
 */
void LDALayer::load(FILE *file)
{
	this->W = m_fread(file);
}

/**
 * Print information about an LDA layer.
 */
void LDALayer::print()
{
	log(LL_VERBOSE, "LDA\n");
	log(LL_VERBOSE, "  %-20s  %10d\n", "n1", this->n1);
	log(LL_VERBOSE, "  %-20s  %10d\n", "n2", this->n2);
}
