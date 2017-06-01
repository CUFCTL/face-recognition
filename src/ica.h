/**
 * @file ica.h
 *
 * Interface definitions for the ICA feature layer.
 */
#ifndef ICA_H
#define ICA_H

#include "feature.h"

typedef enum {
	ICA_NONL_NONE,
	ICA_NONL_POW3,
	ICA_NONL_TANH,
	ICA_NONL_GAUSS
} ica_nonl_t;

class ICALayer : public FeatureLayer {
private:
	int n1;
	int n2;
	ica_nonl_t nonl;
	int max_iter;
	precision_t eps;

	Matrix fpica(const Matrix& X, const Matrix& W_z);

public:
	Matrix W;

	ICALayer(int n1, int n2, ica_nonl_t nonl, int max_iter, precision_t eps);

	void compute(const Matrix& X, const std::vector<DataEntry>& y, int c);
	Matrix project(const Matrix& X);

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print();
};

#endif
