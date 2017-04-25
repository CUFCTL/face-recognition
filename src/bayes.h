/**
 * @file bayes.h
 *
 * Interface definitions for the Bayes classifier.
 */
#ifndef BAYES_H
#define BAYES_H

#include <vector>
#include "dataset.h"
#include "matrix.h"

typedef struct {
	int id;
	matrix_t *entries;
	matrix_t *mu; // mu -> mean vector of each class
	matrix_t *sigma; // sigma -> covariance matrix of each class
} bayes_params_t;

// function decs
data_label_t ** bayesian(matrix_t *X, matrix_t *X_test, data_entry_t *Y, int num_classes);
float calc_bayes_prob(matrix_t *v_test, matrix_t *X_u, matrix_t *X_cov);
int argmax(float *X, int size);

#endif
