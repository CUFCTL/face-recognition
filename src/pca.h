/**
 * @file pca.h
 *
 * Interface definitions for the PCA feature layer.
 */
#ifndef PCA_H
#define PCA_H

#include "feature.h"

class PCALayer : public FeatureLayer {
private:
	int n1;

public:
	Matrix W;
	Matrix D;

	PCALayer(int n1);

	void compute(const Matrix& X, const std::vector<DataEntry>& y, int c);
	Matrix project(const Matrix& X);

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print();
};

#endif
