/**
 * @file identity.h
 *
 * Interface definitions for the identity feature layer.
 */
#ifndef IDENTITY_H
#define IDENTITY_H

#include "feature.h"

class IdentityLayer : public FeatureLayer {
public:
	matrix_t * compute(matrix_t *X, const std::vector<data_entry_t>& y, int c);
	matrix_t * project(matrix_t *X);

	void save(FILE *file);
	void load(FILE *file);

	void print();
};

#endif
