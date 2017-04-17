/**
 * @file model.h
 *
 * Interface definitions for the model type.
 */
#ifndef MODEL_H
#define MODEL_H

#include "ica.h"
#include "image_entry.h"
#include "lda.h"
#include "knn.h"
#include "matrix.h"
#include "pca.h"
#include "bayes.h"

typedef struct {
	pca_params_t pca;
	lda_params_t lda;
	ica_params_t ica;
	knn_params_t knn;
} model_params_t;

typedef struct {
	bool enabled;
	const char *name;
	matrix_t *W;
	matrix_t *P;
} model_algorithm_t;

typedef struct {
	// hyperparameters
	model_params_t params;

	// input
	int num_entries;
	image_entry_t *entries;
	int num_labels;
	image_label_t *labels;
	matrix_t *mean_face;

	// algorithms
	model_algorithm_t pca;
	model_algorithm_t lda;
	model_algorithm_t ica;
} model_t;

model_t * model_construct(bool pca, bool lda, bool ica, model_params_t params);
void model_destruct(model_t *model);

void model_train(model_t *model, const char *path);
void model_save(model_t *model, const char *path);
void model_load(model_t *model, const char *path);
void model_predict(model_t *model, const char *path);

#endif
