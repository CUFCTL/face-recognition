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
	void compute(const Matrix& X, const std::vector<data_entry_t>& y, int c);
	Matrix project(const Matrix& X);

	void save(FILE *file);
	void load(FILE *file);

	void print();
};

#endif
