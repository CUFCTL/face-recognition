/**
 * @file knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <stdlib.h>
#include "knn.h"
#include "math_helper.h"

typedef struct {
	char *label;
	precision_t dist;
} neighbor_t;

/**
 * Print a list of neighbors.
 *
 * @param neighbors
 * @param num
 */
void debug_print_neighbors(neighbor_t *neighbors, int num)
{
	int i;
	for ( i = 0; i < num; i++) {
		printf("%8s  %f\n", neighbors[i].label, neighbors[i].dist);
	}
	putchar('\n');
}

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

	if ( n1->dist < n2->dist ) {
		return -1;
	}
	else if ( n1->dist > n2->dist ) {
		return 1;
	}

	return 0;
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
 * @param params
 * @param X
 * @param Y
 * @param X_test
 * @param i
 * @return predicted label of the test observation
 */
char * kNN(knn_params_t *params, matrix_t *X, const std::vector<data_entry_t>& Y, matrix_t *X_test, int i)
{
	// compute distance between X_test_i and each observation in X
	int num_neighbors = X->cols;
	neighbor_t *neighbors = (neighbor_t *)malloc(num_neighbors * sizeof(neighbor_t));

	int j;
	for ( j = 0; j < num_neighbors; j++ ) {
		neighbors[j].label = Y[j].label;
		neighbors[j].dist = params->dist(X_test, i, X, j);
	}

	// sort the neighbors by distance
	qsort(neighbors, num_neighbors, sizeof(neighbor_t), kNN_compare);

	// determine the mode of the k nearest labels
	char *nearest = (char *)mode(neighbors, params->k, sizeof(neighbor_t), kNN_identify);

	free(neighbors);

	return nearest;
}
