/**
 * @file bayes.cpp
 *
 * Implementation of the naive Bayes classifier.
 */
#include <stdlib.h>
#include "bayes.h"
#include "lda.h"
#include "logger.h"
#include "matrix_utils.h"

/**
 * Construct a Bayes classifier.
 */
BayesLayer::BayesLayer()
{
}

/**
 * Compute the probability of a class for a
 * feature vector using the Bayes discriminant
 * function:
 *
 * g_i'(x) = -1/2 * (x - mu_i)' * S_i^-1 * (x - mu_i)
 */
precision_t bayes_prob(matrix_t *x, matrix_t *mu, matrix_t *S_inv)
{
	m_subtract_columns(x, mu);

	matrix_t *p_temp1 = m_product("p_temp1", x, S_inv, true, false);
	matrix_t *p_temp2 = m_product("p_temp2", p_temp1, x, false, false);

	precision_t p = -0.5f * p_temp2->data[0];

	// cleanup
	m_free(p_temp1);
	m_free(p_temp2);

	return p;
}

/**
 * Classify an observation using naive Bayes.
 *
 * @param X
 * @param Y
 * @param C
 * @param X_test
 * @return predicted labels of the test observations
 */
std::vector<data_label_t> BayesLayer::predict(matrix_t *X, const std::vector<data_entry_t>& Y, const std::vector<data_label_t>& C, matrix_t *X_test)
{
	int num_classes = C.size();
	matrix_t **X_c = m_copy_classes(X, Y, num_classes);
	matrix_t **U = m_class_means(X_c, num_classes);
	matrix_t **S = m_class_scatters(X_c, U, num_classes);

	// compute inverses of each S_i
	matrix_t **S_inv = (matrix_t **) malloc(num_classes * sizeof(matrix_t *));

	int i, j;
	for ( i = 0; i < num_classes; i++ ) {
		S_inv[i] = m_inverse("S_i_inv", S[i]);
	}

	// compute label for each test vector
	std::vector<data_label_t> Y_pred;

	for ( i = 0; i < X_test->cols; i++ ) {
		matrix_t *probs = m_initialize("probs", num_classes, 1);

		for ( j = 0; j < num_classes; j++ ) {
			matrix_t *x_test = m_copy_columns("x_test", X_test, i, i + 1);

			probs->data[j] = bayes_prob(x_test, U[j], S_inv[j]);

			m_free(x_test);
		}

		int index = m_argmax(probs);

		Y_pred.push_back(C[index]);

		m_free(probs);
	}

	// cleanup
	for ( i = 0; i < num_classes; i++ ) {
		m_free(X_c[i]);
		m_free(U[i]);
		m_free(S[i]);
		m_free(S_inv[i]);
	}
	free(X_c);
	free(U);
	free(S);
	free(S_inv);

	return Y_pred;
}

/**
 * Print information about a Bayes classifier.
 */
void BayesLayer::print()
{
	log(LL_VERBOSE, "Bayes\n");
}
