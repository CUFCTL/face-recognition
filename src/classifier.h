/**
 * @file classifier.h
 *
 * Interface definitions for the abstract classifier layer.
 */
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <stdio.h>
#include <vector>
#include "dataset.h"
#include "matrix.h"

class ClassifierLayer {
public:
	virtual ~ClassifierLayer() {};

	virtual char ** predict(
		matrix_t *X,
		const std::vector<data_entry_t>& Y,
		const std::vector<data_label_t>& C,
		matrix_t *X_test
	) = 0;

	virtual void print() = 0;
};

#endif
