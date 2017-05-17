/**
 * @file model.h
 *
 * Interface definitions for the model type.
 */
#ifndef MODEL_H
#define MODEL_H

#include "classifier.h"
#include "dataset.h"
#include "feature.h"
#include "matrix.h"

typedef struct {
	float accuracy;
	float train_time;
	float test_time;
} model_stats_t;

class Model {
private:
	// feature layer
	FeatureLayer *feature;
	Matrix P;

	// classifier layer
	ClassifierLayer *classifier;

	// input data
	Dataset train_set;
	Matrix mean;

	// performance, accuracy stats
	model_stats_t stats;

public:
	Model(FeatureLayer *feature, ClassifierLayer *classifier);
	~Model();

	void train(const Dataset& train_set);
	void save(const char *path);
	void load(const char *path);
	std::vector<data_label_t> predict(const Dataset& test_set);
	void validate(const Dataset& test_set, const std::vector<data_label_t>& Y_pred);
	void print_stats();
};

#endif
