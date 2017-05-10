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

	std::vector<data_label_t> predict(
		matrix_t *X,
		const std::vector<data_entry_t>& Y,
		const std::vector<data_label_t>& C,
		matrix_t *X_test
	);

	void print();
};

#endif
