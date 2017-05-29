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
	Matrix W;

	LDALayer(int n1, int n2);

	void compute(const Matrix& X, const std::vector<data_entry_t>& y, int c);
	Matrix project(const Matrix& X);

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print();
};

#endif
