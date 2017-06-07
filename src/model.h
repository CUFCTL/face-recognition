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
	float error_rate;
	float train_time;
	float test_time;
} model_stats_t;

class Model {
private:
	// input data
	Dataset _train_set;
	Matrix _mean;

	// feature layer
	FeatureLayer *_feature;
	Matrix _P;

	// classifier layer
	ClassifierLayer *_classifier;

	// performance, accuracy stats
	model_stats_t _stats;

public:
	Model(FeatureLayer *feature, ClassifierLayer *classifier);
	~Model();

	void save(const std::string& path);
	void load(const std::string& path);

	void train(const Dataset& train_set);
	std::vector<DataLabel> predict(const Dataset& test_set);
	void validate(const Dataset& test_set, const std::vector<DataLabel>& Y_pred);

	void print_stats();
};

#endif
