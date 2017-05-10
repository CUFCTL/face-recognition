/**
 * @file knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <stdlib.h>
#include "knn.h"
#include "logger.h"
#include "math_utils.h"

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
 * Construct a kNN classifier.
 *
 * @param k
 * @param dist
 */
KNNLayer::KNNLayer(int k, dist_func_t dist)
{
	this->k = k;
	this->dist = dist;
}

/**
 * Classify an observation using k-nearest neighbors.
 *
 * @param X
 * @param Y
 * @param C
 * @param X_test
 * @return predicted labels of the test observations
 */
char ** KNNLayer::predict(matrix_t *X, const std::vector<data_entry_t>& Y, const std::vector<data_label_t>& C, matrix_t *X_test)
{
	char **Y_pred = (char **)malloc(X_test->cols * sizeof(char *));

	int i;
	for ( i = 0; i < X_test->cols; i++ ) {
		// compute distance between X_test_i and each observation in X
		int num_neighbors = X->cols;
		neighbor_t *neighbors = (neighbor_t *)malloc(num_neighbors * sizeof(neighbor_t));

		int j;
		for ( j = 0; j < num_neighbors; j++ ) {
			neighbors[j].label = Y[j].label;
			neighbors[j].dist = this->dist(X_test, i, X, j);
		}

		// sort the neighbors by distance
		qsort(neighbors, num_neighbors, sizeof(neighbor_t), kNN_compare);

		// determine the mode of the k nearest labels
		Y_pred[i] = (char *)mode(neighbors, this->k, sizeof(neighbor_t), kNN_identify);

		// cleanup
		free(neighbors);
	}

	return Y_pred;
}

/**
 * Print information about a kNN classifier.
 */
void KNNLayer::print()
{
	const char *dist_name = "";

	if ( this->dist == m_dist_COS ) {
		dist_name = "COS";
	}
	else if ( this->dist == m_dist_L1 ) {
		dist_name = "L1";
	}
	else if ( this->dist == m_dist_L2 ) {
		dist_name = "L2";
	}

	log(LL_VERBOSE, "kNN\n");
	log(LL_VERBOSE, "  %-20s  %10d\n", "k", this->k);
	log(LL_VERBOSE, "  %-20s  %10s\n", "dist", dist_name);
}
