/**
 * @file bayes.cpp
 *
 * Implementation of the Bayes classifier.
 */
#include <stdlib.h>
#include <string.h>
#include "lda.h"
#include "bayes.h"

data_label_t ** bayesian(matrix_t *X, matrix_t *X_test, data_entry_t *Y, int num_classes)
{
	unsigned int i, j, id;
	float probs[num_classes];

	matrix_t **X_c  = m_copy_classes(X, Y, num_classes);
	matrix_t **X_u  = m_class_means(X_c, num_classes);
	matrix_t *X_cov = m_scatter_between(X_c, X_u, num_classes);

	// allocate space for labels to be returned
	data_label_t **labels = (data_label_t **)malloc(X_test->cols * sizeof(data_label_t *));
	for (i = 0; i < (unsigned int) X_test->cols; i++)
	{
		labels[i] = (data_label_t *)malloc(sizeof(data_label_t));
	}

	for (i = 0; i < (unsigned int) X_test->cols; i++)
	{
		matrix_t *test_vec = m_copy_columns(X_test->name, X_test, i, i + 1);

		for (j = 0; j < num_classes; j++)
		{
			probs[j] = calc_bayes_prob(test_vec, X_u[j], X_cov);
		}
		char str[5];
		labels[i]->id   = argmax(probs, num_classes);
		labels[i]->name = sprintf(str, "s%d", labels[i]->id);

		m_free(test_vec);
	}

	return NULL;
}


// calculate the probability using the Bayesian discriminant function
// prob = -0.5 * (v_test - mu) * inv(sigma) * (v_test - mu)'
float calc_bayes_prob(matrix_t *v_test, matrix_t *X_u, matrix_t *X_cov)
{
	float prob = -9999999;

	m_subtract_columns(v_test, X_u);
	m_elem_mult(v_test, -0.5);

	matrix_t * sigma_inv = m_inverse(X_cov->name, X_cov);

	matrix_t * v_mult_sig_inv = m_product(v_test->name, v_test, sigma_inv, true, false);

	matrix_t * final = m_product("prob_m", v_mult_sig_inv, v_test, false, false);

	prob = final->data[0];

	// cleanup
	m_free(sigma_inv);
	m_free(v_mult_sig_inv);
	m_free(final);

	return prob;
}

int argmax(float *X, int size)
{
	int i, idx = 0;
	float max = X[idx];

	for (i = 0; i < size; i++)
	{
		if (X[i] > max)
		{
			max = X[i];
			idx = i;
		}
	}

	return idx;
}
