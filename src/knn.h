/**
 * @file knn.h
 *
 * Interface definitions for the kNN layer.
 */
#ifndef KNN_H
#define KNN_H

#include "image_entry.h"
#include "matrix.h"

typedef struct {
	int k;
	dist_func_t dist;
} knn_params_t;

image_label_t * kNN(knn_params_t *params, matrix_t *X, image_entry_t *Y, matrix_t *X_test, int i);

#endif
