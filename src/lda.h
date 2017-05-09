/**
 * @file lda.h
 *
 * Interface definitions for the LDA feature layer.
 */
#ifndef LDA_H
#define LDA_H

#include "feature.h"

matrix_t ** m_copy_classes(matrix_t *X, const std::vector<data_entry_t>& y, int c);
matrix_t ** m_class_means(matrix_t **X_c, int c);
matrix_t * m_scatter_between(matrix_t **X_c, matrix_t **U, int c);
matrix_t * m_scatter_within(matrix_t **X_c, matrix_t **U, int c);

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
