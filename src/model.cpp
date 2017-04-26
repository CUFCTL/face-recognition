/**
 * @file model.cpp
 *
 * Implementation of the model type.
 */
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image.h"
#include "logger.h"
#include "model.h"
#include "timer.h"

/**
 * Construct a model.
 *
 * @param feature
 * @param classifier
 * @param params
 * @return pointer to new model
 */
model_t * model_construct(feature_type_t feature, classifier_type_t classifier, model_params_t params)
{
	model_t *model = (model_t *)malloc(sizeof(model_t));
	model->params = params;
	model->feature = feature;
	model->W = NULL;
	model->P = NULL;
	model->classifier = classifier;
	model->dataset = NULL;

	// log hyperparameters
	int len = 20;

	log(LL_VERBOSE, "Hyperparameters\n");

	if ( model->feature == FEATURE_PCA ) {
		log(LL_VERBOSE, "PCA\n");
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "n1", model->params.pca.n1);
	}
	else if ( model->feature == FEATURE_LDA ) {
		log(LL_VERBOSE, "LDA\n");
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "n1", model->params.lda.n1);
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "n2", model->params.lda.n2);
	}
	else if ( model->feature == FEATURE_ICA ) {
		log(LL_VERBOSE, "ICA\n");
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "n1", model->params.ica.n1);
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "n2", model->params.ica.n2);
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "max_iterations", model->params.ica.max_iterations);
		log(LL_VERBOSE, "  %-*s  %10f\n", len, "epsilon", model->params.ica.epsilon);
	}

	if ( model->classifier == CLASSIFIER_KNN ) {
		log(LL_VERBOSE, "kNN\n");
		log(LL_VERBOSE, "  %-*s  %10d\n", len, "k", model->params.knn.k);
		log(LL_VERBOSE, "  %-*s  %10s\n", len, "dist", model->params.knn.dist_name);
	}
	else if ( model->classifier == CLASSIFIER_BAYES ) {
		log(LL_VERBOSE, "Bayes\n");
	}

	log(LL_VERBOSE, "\n");

	return model;
}

/**
 * Destruct a model.
 *
 * @param model
 */
void model_destruct(model_t *model)
{
	dataset_destruct(model->dataset);

	m_free(model->mean);

	m_free(model->W);
	m_free(model->P);

	free(model);
}

/**
 * Perform training on a training set.
 *
 * @param model
 * @param train_set
 */
void model_train(model_t *model, dataset_t *train_set)
{
	timer_push("Training");

	model->dataset = train_set;

	log(LL_VERBOSE, "Training set: %d samples, %d classes\n",
		train_set->num_entries,
		train_set->num_labels);

	// get data matrix X
	matrix_t *X = dataset_load(train_set);

	// subtract mean from X
	model->mean = m_mean_column("m", X);
	m_subtract_columns(X, model->mean);

	// compute features from X
	if ( model->feature == FEATURE_PCA ) {
		model->W = PCA(&model->params.pca, X, NULL);
		model->P = m_product("P_pca", model->W, X, true, false);
	}
	else if ( model->feature == FEATURE_LDA ) {
		model->W = LDA(&model->params.lda, X, train_set->entries, train_set->num_labels);
		model->P = m_product("P_lda", model->W, X, true, false);
	}
	else if ( model->feature == FEATURE_ICA ) {
		model->W = ICA(&model->params.ica, X);
		model->P = m_product("P_ica", model->W, X, true, false);
	}

	model->stats.train_time = timer_pop();

	log(LL_VERBOSE, "\n");

	// cleanup
	m_free(X);
}

/**
 * Save a model to a file.
 *
 * @param model
 * @param path
 */
void model_save(model_t *model, const char *path)
{
	FILE *file = fopen(path, "w");

	dataset_fwrite(model->dataset, file);

	m_fwrite(file, model->mean);

	m_fwrite(file, model->W);
	m_fwrite(file, model->P);

	fclose(file);
}

/**
 * Load a model from a file.
 *
 * @param model
 * @param path
 */
void model_load(model_t *model, const char *path)
{
	FILE *file = fopen(path, "r");

	model->dataset = dataset_fread(file);

	model->mean = m_fread(file);

	model->W = m_fread(file);
	model->P = m_fread(file);

	fclose(file);
}

/**
 * Perform recognition on a test set.
 *
 * @param model
 * @param test_set
 */
data_label_t **model_predict(model_t *model, dataset_t *test_set)
{
	timer_push("Recognition");

	log(LL_VERBOSE, "Test set: %d samples, %d classes\n",
		test_set->num_entries,
		test_set->num_labels);

	// compute projected test images
	matrix_t *X_test = dataset_load(test_set);
	m_subtract_columns(X_test, model->mean);

	matrix_t *P_test = m_product("P_test", model->W, X_test, true, false);

	// compute predicted labels
	data_label_t **pred_labels = (data_label_t **)malloc(test_set->num_entries * sizeof(data_label_t *));

	if ( model->classifier == CLASSIFIER_KNN ) {
		int i;
		for ( i = 0; i < test_set->num_entries; i++ ) {
			pred_labels[i] = kNN(&model->params.knn, model->P, model->dataset->entries, P_test, i);
		}
	}
	else if ( model->classifier == CLASSIFIER_BAYES ) {
		pred_labels = bayesian(model->P, P_test, model->dataset->labels, model->dataset->entries, test_set->num_labels);
	}

	model->stats.test_time = timer_pop();

	log(LL_VERBOSE, "\n");

	// cleanup
	m_free(X_test);
	m_free(P_test);

	return pred_labels;
}

/**
 * Validate a set of predicted labels against the ground truth.
 *
 * @param model
 * @param test_set
 * @param pred_labels
 */
void model_validate(model_t *model, dataset_t *test_set, data_label_t **pred_labels)
{
	// compute accuracy
	int num_correct = 0;

	int i;
	for ( i = 0; i < test_set->num_entries; i++ ) {
		if ( strcmp(pred_labels[i]->name, test_set->entries[i].label->name) == 0 ) {
			num_correct++;
		}
	}

	model->stats.accuracy = 100.0f * num_correct / test_set->num_entries;

	// print results
	log(LL_VERBOSE, "Results\n");

	for ( i = 0; i < test_set->num_entries; i++ ) {
		data_label_t *pred_label = pred_labels[i];
		data_entry_t *entry = &test_set->entries[i];

		const char *s = (strcmp(pred_label->name, entry->label->name) != 0)
			? "(!)"
			: "";

		log(LL_VERBOSE, "%-10s -> %-4s %s\n",
			basename(entry->name),
			pred_label->name, s);
	}

	log(LL_VERBOSE, "%d / %d matched, %.2f%%\n",
		num_correct,
		test_set->num_entries,
		model->stats.accuracy);
	log(LL_VERBOSE, "\n");
}

/**
 * Print a model's performance / accuracy stats.
 *
 * @param model
 */
void model_print_stats(model_t *model)
{
	printf("%10.2f  %10.3f  %10.3f\n",
		model->stats.accuracy,
		model->stats.train_time,
		model->stats.test_time);
}
