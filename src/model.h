/**
 * @file model.h
 *
 * Interface definitions for the model type.
 */
#ifndef MODEL_H
#define MODEL_H

#include "bayes.h"
#include "dataset.h"
#include "ica.h"
#include "lda.h"
#include "knn.h"
#include "matrix.h"
#include "pca.h"

typedef enum {
	FEATURE_NONE,
	FEATURE_PCA,
	FEATURE_LDA,
	FEATURE_ICA
} feature_type_t;

typedef enum {
	CLASSIFIER_NONE,
	CLASSIFIER_KNN,
	CLASSIFIER_BAYES
} classifier_type_t;

typedef struct {
	pca_params_t pca;
	lda_params_t lda;
	ica_params_t ica;
	knn_params_t knn;
} model_params_t;

typedef struct {
	float accuracy;
	float train_time;
	float test_time;
} model_stats_t;

class Model {
private:
	// hyperparameters
	model_params_t params;

	// feature layer
	feature_type_t feature;
	matrix_t *W;
	matrix_t *P;

	// classifier layer
	classifier_type_t classifier;

	// input data
	Dataset train_set;
	matrix_t *mean;

	// performance, accuracy stats
	model_stats_t stats;

public:
	Model(feature_type_t feature, classifier_type_t classifier, model_params_t params);
	~Model();

	void train(const Dataset& train_set);
	void save(const char *path);
	void load(const char *path);
	char ** predict(const Dataset& test_set);
	void validate(const Dataset& test_set, char **pred_labels);
	void print_stats();
};

#endif
