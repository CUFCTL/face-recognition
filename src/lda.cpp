/**
 * @file lda.cpp
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include <stdlib.h>
#include "lda.h"
#include "logger.h"
#include "matrix_utils.h"
#include "pca.h"
#include "timer.h"

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
