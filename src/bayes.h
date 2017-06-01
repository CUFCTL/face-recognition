/**
 * @file bayes.h
 *
 * Interface definitions for the naive Bayes classifier.
 */
#ifndef BAYES_H
#define BAYES_H

#include "classifier.h"

class BayesLayer : public ClassifierLayer {
public:
	BayesLayer();

	std::vector<DataLabel> predict(
		const Matrix& X,
		const std::vector<DataEntry>& Y,
		const std::vector<DataLabel>& C,
		const Matrix& X_test
	);

	void print();
};

#endif
