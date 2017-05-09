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

	matrix_t * fpica(matrix_t *X, matrix_t *W_z);

public:
	matrix_t *W;

	ICALayer(int n1, int n2, ica_nonl_t nonl, int max_iter, precision_t eps);
	~ICALayer();

	matrix_t * compute(matrix_t *X, const std::vector<data_entry_t>& y, int c);
	matrix_t * project(matrix_t *X);

	void save(FILE *file);
	void load(FILE *file);

	void print();
};

#endif
