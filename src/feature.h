/**
 * @file feature.h
 *
 * Interface definitions for the abstract feature layer.
 */
#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include "dataset.h"
#include "matrix.h"

class FeatureLayer {
public:
	virtual ~FeatureLayer() {};

	virtual void compute(const Matrix& X, const std::vector<DataEntry>& y, int c) = 0;
	virtual Matrix project(const Matrix& X) = 0;

	virtual void save(std::ofstream& file) = 0;
	virtual void load(std::ifstream& file) = 0;

	virtual void print() = 0;
};

#endif
