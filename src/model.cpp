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

const feature_layer_t FEATURES[] = {
	{
		FEATURE_PCA,
		"PCA",
		NULL, NULL
	},
	{
		FEATURE_LDA,
		"LDA",
		NULL, NULL
	},
	{
		FEATURE_ICA,
		"ICA",
		NULL, NULL
	}
};
const int NUM_FEATURES = sizeof(FEATURES) / sizeof(feature_layer_t);

/**
 * Map a collection of images to column vectors.
 *
 * The image matrix has size m x n, where m is the number of
 * pixels in each image and n is the number of images. The
 * images must all have the same size.
 *
 * @param entries
 * @param num_entries
 * @return pointer to image matrix
 */
matrix_t * get_image_matrix(image_entry_t *entries, int num_entries)
{
	// get the image size from the first image
	image_t *image = image_construct();
	image_read(image, entries[0].name);

	// construct image matrix
	matrix_t *X = m_initialize("X", image->channels * image->height * image->width, num_entries);

	// map each image to a column vector
	m_image_read(X, 0, image);

	int i;
	for ( i = 1; i < num_entries; i++ ) {
		image_read(image, entries[i].name);
		m_image_read(X, i, image);
	}

	image_destruct(image);

	return X;
}

/**
 * Construct a model.
 *
 * @param feature_type
 * @param params
 * @return pointer to new model
 */
model_t * model_construct(feature_type_t feature_type, model_params_t params)
{
	model_t *model = (model_t *)calloc(1, sizeof(model_t));
	model->params = params;

	int i;
	for ( i = 0; i < NUM_FEATURES; i++ ) {
		if ( FEATURES[i].type == feature_type ) {
			model->feature_layer = FEATURES[i];
		}
	}

	if ( LOGGER(LL_VERBOSE) ) {
		int len = 20;

		printf("Hyperparameters\n");

		if ( model->feature_layer.type == FEATURE_PCA ) {
			printf("PCA\n");
			printf("  %-*s  %10d\n", len, "n1", model->params.pca.n1);
		}
		else if ( model->feature_layer.type == FEATURE_LDA ) {
			printf("LDA\n");
			printf("  %-*s  %10d\n", len, "n1", model->params.lda.n1);
			printf("  %-*s  %10d\n", len, "n2", model->params.lda.n2);
		}
		else if ( model->feature_layer.type == FEATURE_ICA ) {
			printf("ICA\n");
			printf("  %-*s  %10d\n", len, "n1", model->params.ica.n1);
			printf("  %-*s  %10d\n", len, "n2", model->params.ica.n2);
			printf("  %-*s  %10d\n", len, "max_iterations", model->params.ica.max_iterations);
			printf("  %-*s  %10f\n", len, "epsilon", model->params.ica.epsilon);
		}

		printf("kNN\n");
		printf("  %-*s  %10d\n", len, "k", model->params.knn.k);
		putchar('\n');
	}

	return model;
}

/**
 * Destruct a model.
 *
 * @param model
 */
void model_destruct(model_t *model)
{
	// free entries
	int i;
	for ( i = 0; i < model->num_entries; i++ ) {
		free(model->entries[i].name);
	}
	free(model->entries);

	// free labels
	for ( i = 0; i < model->num_labels; i++ ) {
		free(model->labels[i].name);
	}
	free(model->labels);

	// free mean face
	m_free(model->mean);

	// free feature layer
	m_free(model->feature_layer.W);
	m_free(model->feature_layer.P);

	free(model);
}

/**
 * Perform training on a training set.
 *
 * @param model
 * @param path
 */
void model_train(model_t *model, const char *path)
{
	timer_push("Training");

	// get entries, labels
	model->num_entries = get_directory(path, &model->entries, &model->num_labels, &model->labels);

	if ( LOGGER(LL_VERBOSE) ) {
		printf("  Training set: %d samples, %d classes\n", model->num_entries, model->num_labels);
	}

	// get image matrix X
	matrix_t *X = get_image_matrix(model->entries, model->num_entries);

	// subtract mean from X
	model->mean = m_mean_column("m", X);
	m_subtract_columns(X, model->mean);

	// perform feature extraction
	if ( model->feature_layer.type == FEATURE_PCA ) {
		model->feature_layer.W = PCA(&model->params.pca, X, NULL);
		model->feature_layer.P = m_product("P_pca", model->feature_layer.W, X, true, false);
	}
	else if ( model->feature_layer.type == FEATURE_LDA ) {
		model->feature_layer.W = LDA(&model->params.lda, X, model->num_labels, model->entries);
		model->feature_layer.P = m_product("P_lda", model->feature_layer.W, X, true, false);
	}
	else if ( model->feature_layer.type == FEATURE_ICA ) {
		model->feature_layer.W = ICA(&model->params.ica, X);
		model->feature_layer.P = m_product("P_ica", model->feature_layer.W, X, true, false);
	}

	timer_pop();

	// cleanup
	m_free(X);
}

/**
 * Save a model to a data file.
 *
 * @param model
 * @param path
 */
void model_save(model_t *model, const char *path)
{
	FILE *file = fopen(path, "w");

	// save labels
	fwrite(&model->num_labels, sizeof(int), 1, file);

	int i;
	for ( i = 0; i < model->num_labels; i++ ) {
		fwrite(&model->labels[i].id, sizeof(int), 1, file);

		int num = strlen(model->labels[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(model->labels[i].name, sizeof(char), num, file);
	}

	// save entries
	fwrite(&model->num_entries, sizeof(int), 1, file);

	for ( i = 0; i < model->num_entries; i++ ) {
		fwrite(&model->entries[i].label->id, sizeof(int), 1, file);

		int num = strlen(model->entries[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(model->entries[i].name, sizeof(char), num, file);
	}

	// save mean face
	m_fwrite(file, model->mean);

	// save feature layer
	fwrite(&model->feature_layer.type, sizeof(feature_type_t), 1, file);
	m_fwrite(file, model->feature_layer.W);
	m_fwrite(file, model->feature_layer.P);

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

	// read labels
	fread(&model->num_labels, sizeof(int), 1, file);

	model->labels = (image_label_t *)malloc(model->num_labels * sizeof(image_label_t));

	int i;
	for ( i = 0; i < model->num_labels; i++ ) {
		fread(&model->labels[i].id, sizeof(int), 1, file);

		int num;
		fread(&num, sizeof(int), 1, file);

		model->labels[i].name = (char *)malloc(num * sizeof(char));
		fread(model->labels[i].name, sizeof(char), num, file);
	}

	// read entries
	fread(&model->num_entries, sizeof(int), 1, file);

	model->entries = (image_entry_t *)malloc(model->num_entries * sizeof(image_entry_t));

	for ( i = 0; i < model->num_entries; i++ ) {
		int label_id;
		fread(&label_id, sizeof(int), 1, file);

		model->entries[i].label = &model->labels[label_id];

		int num;
		fread(&num, sizeof(int), 1, file);

		model->entries[i].name = (char *)malloc(num * sizeof(char));
		fread(model->entries[i].name, sizeof(char), num, file);
	}

	// read mean face
	model->mean = m_fread(file);

	// read feature layer
	feature_type_t feature_type;
	fread(&feature_type, sizeof(feature_type_t), 1, file);

	for ( i = 0; i < NUM_FEATURES; i++ ) {
		if ( FEATURES[i].type == feature_type ) {
			model->feature_layer = FEATURES[i];
		}
	}

	model->feature_layer.W = m_fread(file);
	model->feature_layer.P = m_fread(file);

	fclose(file);
}

/**
 * Perform recognition on a test set.
 *
 * @param model
 * @param path
 */
void model_predict(model_t *model, const char *path)
{
	timer_push("Recognition");

	// get entries, labels
	image_label_t *labels;
	int num_labels;

	image_entry_t *entries;
	int num_entries = get_directory(path, &entries, &num_labels, &labels);

	if ( LOGGER(LL_VERBOSE) ) {
		printf("  Test set: %d samples, %d classes\n", num_entries, num_labels);
	}

	// get image matrix X_test
	matrix_t *X_test = get_image_matrix(entries, num_entries);

	// subtract training set mean from X_test
	m_subtract_columns(X_test, model->mean);

	// compute projected test images
	matrix_t *P_test = m_product("P_test", model->feature_layer.W, X_test, true, false);

	// compute labels for each test image
	image_label_t **rec_labels = (image_label_t **)malloc(num_entries * sizeof(image_label_t *));

	int i;
	for ( i = 0; i < num_entries; i++ ) {
		rec_labels[i] = kNN(&model->params.knn, model->feature_layer.P, model->entries, P_test, i);
	}

	// debugging bayesian classifier
	// image_label_t **bayes_rec_labels = bayesian(algo->P, P_test, model->num_entries, model->num_labels);

	// compute accuracy
	int num_correct = 0;

	for ( i = 0; i < num_entries; i++ ) {
		if ( strcmp(rec_labels[i]->name, entries[i].label->name) == 0 ) {
			num_correct++;
		}
	}

	float accuracy = 100.0f * num_correct / num_entries;

	// print results
	if ( LOGGER(LL_VERBOSE) ) {
		printf("  %s\n", model->feature_layer.name);

		for ( i = 0; i < num_entries; i++ ) {
			const char *s = (strcmp(rec_labels[i]->name, entries[i].label->name) != 0)
				? "(!)"
				: "";

			printf("    %-10s -> %-4s %s\n", basename(entries[i].name), rec_labels[i]->name, s);
		}

		printf("    %d / %d matched, %.2f%%\n", num_correct, num_entries, accuracy);
		putchar('\n');
	}
	else {
		printf("%.2f\n", accuracy);
	}

	// cleanup
	m_free(P_test);
	free(rec_labels);

	timer_pop();

	// cleanup
	for ( i = 0; i < num_entries; i++ ) {
		free(entries[i].name);
	}
	free(entries);

	for ( i = 0; i < num_labels; i++ ) {
		free(labels[i].name);
	}
	free(labels);

	m_free(X_test);
}
