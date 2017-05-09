/**
 * @file knn.h
 *
 * Interface definitions for the kNN layer.
 */
#ifndef KNN_H
#define KNN_H

#include "dataset.h"
#include "matrix.h"

typedef struct {
	int k;
	dist_func_t dist;
	const char *dist_name;
} knn_params_t;

char * kNN(knn_params_t *params, matrix_t *X, const std::vector<data_entry_t>& Y, matrix_t *X_test, int i);

#endif
