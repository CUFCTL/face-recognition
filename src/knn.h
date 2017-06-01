/**
 * @file knn.h
 *
 * Interface definitions for the kNN classifier.
 */
#ifndef KNN_H
#define KNN_H

#include "classifier.h"
#include "matrix_utils.h"

class KNNLayer : public ClassifierLayer {
private:
	int k;
	dist_func_t dist;

public:
	KNNLayer(int k, dist_func_t dist);

	std::vector<DataLabel> predict(
		const Matrix& X,
		const std::vector<DataEntry>& Y,
		const std::vector<DataLabel>& C,
		const Matrix& X_test
	);

	void print();
};

#endif
