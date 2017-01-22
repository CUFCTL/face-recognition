/**
 * @file knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <stdlib.h>
#include "database.h"

typedef struct {
	image_label_t *label;
	precision_t dist;
} neighbor_t;

typedef struct {
	image_label_t *label;
	int count;
} label_count_t;

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
	// TODO: maybe replace with mode function
	label_count_t *counts = (label_count_t *)calloc(k, sizeof(label_count_t));

	for ( j = 0; j < k; j++ ) {
		int n = 0;
		while ( counts[n].label != NULL && counts[n].label != neighbors[j].label ) {
			n++;
		}

		counts[n].label = neighbors[j].label;
		counts[n].count++;
	}

	label_count_t *max = NULL;

	for ( j = 0; counts[j].label != NULL; j++ ) {
		if ( max == NULL || max->count < counts[j].count ) {
			max = &counts[j];
		}
	}

	return max->label;
}

