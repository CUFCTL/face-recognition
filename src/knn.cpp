/**
 * @file knn.cpp
 *
 * Implementation of the k-nearest neighbors classifier.
 */
#include <algorithm>
#include "knn.h"
#include "logger.h"

typedef struct {
	data_label_t label;
	precision_t dist;
} neighbor_t;

typedef struct {
	data_label_t id;
	int count;
} item_count_t;

/**
 * Print a list of neighbors.
 *
 * @param neighbors
 * @param num
 */
void debug_print_neighbors(const std::vector<neighbor_t>& neighbors)
{
	size_t i;
	for ( i = 0; i < neighbors.size(); i++) {
		printf("%8s  %f\n", neighbors[i].label.c_str(), neighbors[i].dist);
	}
	putchar('\n');
}

/**
 * Comparison function for sorting neighbors.
 *
 * @param a
 * @param b
 */
bool kNN_compare(const neighbor_t& a, const neighbor_t& b)
{
	return (a.dist < b.dist);
}

/**
 * Determine the mode of a list of neighbors.
 *
 * @param items
 */
data_label_t kNN_mode(const std::vector<neighbor_t>& items)
{
	std::vector<item_count_t> counts;

	// compute the frequency of each item in the list
	size_t i;
	for ( i = 0; i < items.size(); i++ ) {
		const data_label_t& id = items[i].label;

		size_t j = 0;
		while ( j < counts.size() && counts[j].id != id ) {
			j++;
		}

		if ( j == counts.size() ) {
			item_count_t count;
			count.id = id;
			count.count = 1;

			counts.push_back(count);
		}
		else {
			counts[j].count++;
		}
	}

	// find the item with the highest frequency
	item_count_t max = counts[0];

	for ( i = 1; i < counts.size(); i++ ) {
		if ( max.count < counts[i].count ) {
			max = counts[i];
		}
	}

	return max.id;
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
std::vector<data_label_t> KNNLayer::predict(const Matrix& X, const std::vector<data_entry_t>& Y, const std::vector<data_label_t>& C, const Matrix& X_test)
{
	std::vector<data_label_t> Y_pred;

	int i;
	for ( i = 0; i < X_test.cols(); i++ ) {
		// compute distance between X_test_i and each X_i
		std::vector<neighbor_t> neighbors;

		int j;
		for ( j = 0; j < X.cols(); j++ ) {
			neighbor_t n;
			n.label = Y[j].label;
			n.dist = this->dist(X_test, i, X, j);

			neighbors.push_back(n);
		}

		// determine the k nearest neighbors
		std::sort(neighbors.begin(), neighbors.end(), kNN_compare);

		neighbors.erase(neighbors.begin() + this->k, neighbors.end());

		// determine the mode of the k nearest labels
		data_label_t y_pred = kNN_mode(neighbors);

		Y_pred.push_back(y_pred);
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

	log(LL_VERBOSE, "kNN");
	log(LL_VERBOSE, "  %-20s  %10d", "k", this->k);
	log(LL_VERBOSE, "  %-20s  %10s", "dist", dist_name);
}
