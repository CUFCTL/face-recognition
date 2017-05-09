/**
 * @file pca.cpp
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "logger.h"
#include "math_helper.h"
#include "pca.h"
#include "timer.h"

/**
 * Construct a PCA layer.
 *
 * @param n1
 */
PCALayer::PCALayer(int n1)
{
	this->n1 = n1;

	this->W = NULL;
	this->D = NULL;
}

/**
 * Destruct a PCA layer.
 */
PCALayer::~PCALayer()
{
	m_free(this->W);
	m_free(this->D);
}

/**
 * Compute the principal components of a matrix X, which
 * consists of observations in rows or columns. The observations
 * should also be mean-subtracted.
 *
 * The principal components of a matrix are the eigenvectors of
 * the covariance matrix.
 *
 * @param X
 * @param y
 * @param c
 * @return principal components of X in columns
 */
matrix_t * PCALayer::compute(matrix_t *X, const std::vector<data_entry_t>& y, int c)
{
	// if n1 = -1, use default value
	int n1 = (this->n1 == -1)
		? min(X->rows, X->cols)
		: this->n1;

	timer_push("PCA");

	if ( X->rows > X->cols ) {
		timer_push("compute surrogate of covariance matrix L");

		matrix_t *L = m_product("L", X, X, true, false);

		timer_pop();

		timer_push("compute eigendecomposition of L");

		matrix_t *V;
		m_eigen("V", "D", L, n1, &V, &this->D);

		timer_pop();

		timer_push("compute principal components");

		this->W = m_product("W_pca", X, V);

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

		m_eigen("W_pca", "D", C, n1, &this->W, &this->D);

		timer_pop();

		// cleanup
		m_free(C);
	}

	timer_pop();

	return this->W;
}

/**
 * Project a matrix X into the feature space of a PCA layer.
 *
 * @param X
 * @return projected matrix
 */
matrix_t * PCALayer::project(matrix_t *X)
{
	return m_product("P", this->W, X, true, false);
}

/**
 * Save a PCA layer to a file.
 */
void PCALayer::save(FILE *file)
{
	m_fwrite(file, this->W);
}

/**
 * Load an PCA layer from a file.
 */
void PCALayer::load(FILE *file)
{
	this->W = m_fread(file);
}

/**
 * Print information about a PCA layer.
 */
void PCALayer::print()
{
	log(LL_VERBOSE, "PCA\n");
	log(LL_VERBOSE, "  %-20s  %10d\n", "n1", this->n1);
}
