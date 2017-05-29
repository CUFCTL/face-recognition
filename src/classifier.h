/**
 * @file classifier.h
 *
 * Interface definitions for the abstract classifier layer.
 */
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include "dataset.h"
#include "matrix.h"

class ClassifierLayer {
public:
	virtual ~ClassifierLayer() {};

	virtual std::vector<data_label_t> predict(
		const Matrix& X,
		const std::vector<data_entry_t>& Y,
		const std::vector<data_label_t>& C,
		const Matrix& X_test
	) = 0;

	virtual void print() = 0;
};

#endif
