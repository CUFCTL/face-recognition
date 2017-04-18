/**
 * @file bayes.h
 *
 * Interface definitions for the Bayes classifier.
 */
#ifndef BAYES_H
#define BAYES_H

#include <vector>
#include "image_entry.h"
#include "matrix.h"

typedef struct {
	int id;
	matrix_t *entries;
	matrix_t *mu; // mu -> mean vector of each class
	matrix_t *sigma; // sigma -> covariance matrix of each class
} bayes_params_t;

// function decs
image_label_t ** bayesian(matrix_t *X, matrix_t *X_test, int num_samples, int num_classes);
float calc_bayes_prob(matrix_t *v_test, bayes_params_t param);
std::vector<bayes_params_t> separate_data(matrix_t *X, int num_samples, int num_classes);
matrix_t * class_mean(matrix_t *X);
matrix_t * class_covariance(matrix_t *X);

#endif
