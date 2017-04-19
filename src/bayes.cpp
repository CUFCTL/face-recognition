/**
 * @file bayes.cpp
 *
 * Implementation of the Bayes classifier.
 */
#include <stdlib.h>
#include <string.h>
#include "bayes.h"

data_label_t ** bayesian(matrix_t *X, matrix_t *X_test, int num_samples, int num_classes)
{

	unsigned int i, j;
	std::vector<float> probs(num_classes);

	std::vector<bayes_params_t> param_list = separate_data(X, num_samples, num_classes);

	for (i = 0; i < param_list.size(); i++)
	{
		param_list[i].mu = class_mean(param_list[i].entries);
		param_list[i].sigma = class_covariance(param_list[i].entries);
	}

	for (i = 0; i < (unsigned int) X_test->cols; i++)
	{
		matrix_t *test_vec = m_copy_columns(X_test->name, X_test, i, i + 1);

		for (j = 0; j < param_list.size(); j++)
		{
			probs[j] = calc_bayes_prob(test_vec, param_list[j]);
		}

		m_free(test_vec);
	}

	return NULL;
}


// calculate the probability using the Bayesian discriminant function
// prob = -0.5 * (v_test - mu) * inv(sigma) * (v_test - mu)'
float calc_bayes_prob(matrix_t *v_test, bayes_params_t param)
{
	float prob = -999.999;

	m_subtract_columns(v_test, param.mu);
	m_elem_mult(v_test, -0.5);
	
	matrix_t * sigma_inv = m_inverse(param.sigma->name, param.sigma);
	
	matrix_t * v_mult_sig_inv = m_product(v_test->name, v_test, sigma_inv, true, false);
	
	matrix_t * final = m_product("prob_m", v_mult_sig_inv, v_test, false, false);

	prob = final->data[0];

	// cleanup
	m_free(sigma_inv);
	m_free(v_mult_sig_inv);
	m_free(final);

	return prob;
}


// separates train data by class
std::vector<bayes_params_t> separate_data(matrix_t *X, int num_samples, int num_classes)
{
	int i = 0;

	// user a vector to collect the params of each class
	std::vector<bayes_params_t> params(num_classes);

	for (i = 0; i < num_classes; i++)
	{
		char *name = (char *)malloc(10*sizeof(char));
		sprintf(name, "class_%d", i + 1);

		//params.push_back(bayes_params_t());
		params[i].id = i + 1;
		params[i].entries = m_copy_columns(name, X, (i * (num_samples/num_classes)), (i + 1) * (num_samples/num_classes));
	}

	return params;
}


// calculate mean vector of each class
matrix_t * class_mean(matrix_t *X)
{
	return m_mean_column(X->name, X);
}


// calculate the covariance matrix of each class
matrix_t * class_covariance(matrix_t *X)
{
	return m_product(X->name, X, X, false, true);
}
