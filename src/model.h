/**
 * @file model.h
 *
 * Interface definitions for the model type.
 */
#ifndef MODEL_H
#define MODEL_H

#include "bayes.h"
#include "ica.h"
#include "image_entry.h"
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
	feature_type_t type;
	const char *name;
	matrix_t *W;
	matrix_t *P;
} feature_layer_t;

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
	feature_layer_t feature_layer;

	// classifier layer
	classifier_type_t classifier_type;

	// input
	int num_entries;
	image_entry_t *entries;
	int num_labels;
	image_label_t *labels;
	matrix_t *mean;
} model_t;

model_t * model_construct(feature_type_t feature_type, classifier_type_t classifier_type, model_params_t params);
void model_destruct(model_t *model);

void model_train(model_t *model, const char *path);
void model_save(model_t *model, const char *path);
void model_load(model_t *model, const char *path);
image_label_t ** model_predict(model_t *model, const char *path);
void model_validate(model_t *model, const char *path, image_label_t **pred_labels);

#endif
