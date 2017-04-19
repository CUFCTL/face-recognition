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
	// hyperparameters
	model_params_t params;

	// feature layer
	feature_type_t feature;
	matrix_t *W;
	matrix_t *P;

	// classifier layer
	classifier_type_t classifier;

	// input data
	dataset_t *dataset;
	matrix_t *mean;
} model_t;

model_t * model_construct(feature_type_t feature, classifier_type_t classifier, model_params_t params);
void model_destruct(model_t *model);

void model_train(model_t *model, dataset_t *train_set);
void model_save(model_t *model, const char *path);
void model_load(model_t *model, const char *path);
data_label_t ** model_predict(model_t *model, dataset_t *test_set);
void model_validate(model_t *model, dataset_t *test_set, data_label_t **pred_labels);

#endif
