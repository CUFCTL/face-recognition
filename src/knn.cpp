/**
 * @file knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <stdlib.h>
#include "database.h"
#include "math_helper.h"

typedef struct {
	image_label_t *label;
	precision_t dist;
} neighbor_t;

/**
 * Comparison function for the kNN classifier.
 *
 * @param a
 * @param b
 */
int kNN_compare(const void *a, const void *b)
{
	neighbor_t *n1 = (neighbor_t *)a;
	neighbor_t *n2 = (neighbor_t *)b;

	return (int)(n1->dist - n2->dist);
}

/**
 * Identification function for the kNN classifier.
 *
 * @param a
 */
void * kNN_identify(const void *a)
{
	neighbor_t *n = (neighbor_t *)a;

	return n->label;
}

/**
 * Classify an observation using k-nearest neighbors.
 *
 * @param X
 * @param Y
 * @param X_test
 * @param i
 * @param k
 * @param dist_func
 * @return predicted label of the test observation
 */
image_label_t * kNN(matrix_t *X, image_entry_t *Y, matrix_t *X_test, int i, int k, dist_func_t dist_func)
{
	// compute distance between X_test_i and each observation in X
	neighbor_t *neighbors = (neighbor_t *)malloc(X->cols * sizeof(neighbor_t));

	int j;
	for ( j = 0; j < X->cols; j++ ) {
		neighbors[j].label = Y[j].label;
		neighbors[j].dist = dist_func(X_test, i, X, j);
	}

	// sort the neighbors by distance
	qsort(neighbors, X->cols, sizeof(neighbor_t), kNN_compare);

	// determine the mode of the k nearest labels
	image_label_t *nearest = (image_label_t *)mode(neighbors, k, sizeof(neighbor_t), kNN_identify);

	free(neighbors);

	return nearest;
}