/**
 * @file bayes.cpp
 *
 * Implementation of the naive Bayes classifier.
 */
#include <algorithm>
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
precision_t bayes_prob(Matrix& x, const Matrix& mu, const Matrix& S_inv)
{
	x.subtract_columns(mu);

	Matrix p_temp1 = x.T->product("p_temp1", S_inv);
	Matrix p_temp2 = p_temp1.product("p_temp2", x);

	precision_t p = -0.5f * p_temp2.elem(0, 0);

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
std::vector<data_label_t> BayesLayer::predict(const Matrix& X, const std::vector<data_entry_t>& Y, const std::vector<data_label_t>& C, const Matrix& X_test)
{
	std::vector<Matrix> X_c = m_copy_classes(X, Y, C.size());
	std::vector<Matrix> U = m_class_means(X_c, C.size());
	std::vector<Matrix> S = m_class_scatters(X_c, U, C.size());

	// compute inverses of each S_i
	std::vector<Matrix> S_inv;

	unsigned i, j;
	for ( i = 0; i < C.size(); i++ ) {
		S_inv.push_back(S[i].inverse("S_i_inv"));
	}

	// compute label for each test vector
	std::vector<data_label_t> Y_pred;

	for ( i = 0; i < X_test.cols(); i++ ) {
		std::vector<precision_t> probs;

		for ( j = 0; j < C.size(); j++ ) {
			Matrix x_test("x_test", X_test, i, i + 1);

			probs.push_back(bayes_prob(x_test, U[j], S_inv[j]));
		}

		int index = *max_element(probs.begin(), probs.end());

		Y_pred.push_back(C[index]);
	}

	return Y_pred;
}

/**
 * Print information about a Bayes classifier.
 */
void BayesLayer::print()
{
	log(LL_VERBOSE, "Bayes\n");
}
