/**
 * @file lda.h
 *
 * Interface definitions for the LDA feature layer.
 */
#ifndef LDA_H
#define LDA_H

#include "feature.h"

class LDALayer : public FeatureLayer {
private:
	int n1;
	int n2;

public:
	matrix_t *W;

	LDALayer(int n1, int n2);
	~LDALayer();

	matrix_t * compute(matrix_t *X, const std::vector<data_entry_t>& y, int c);
	matrix_t * project(matrix_t *X);

	void save(FILE *file);
	void load(FILE *file);

	void print();
};

#endif
