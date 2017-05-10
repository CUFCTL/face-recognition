/**
 * @file knn.h
 *
 * Interface definitions for the kNN classifier.
 */
#ifndef KNN_H
#define KNN_H

#include "classifier.h"

class KNNLayer : public ClassifierLayer {
private:
	int k;
	dist_func_t dist;

public:
	KNNLayer(int k, dist_func_t dist);

	std::vector<data_label_t> predict(
		matrix_t *X,
		const std::vector<data_entry_t>& Y,
		const std::vector<data_label_t>& C,
		matrix_t *X_test
	);

	void print();
};

#endif
