/**
 * @file database.h
 *
 * Interface definitions for the database type.
 */
#ifndef DATABASE_H
#define DATABASE_H

#include "ica.h"
#include "image_entry.h"
#include "lda.h"
#include "knn.h"
#include "matrix.h"
#include "pca.h"

typedef struct {
	pca_params_t pca;
	lda_params_t lda;
	ica_params_t ica;
	knn_params_t knn;
} db_params_t;

typedef struct {
	bool train;
	bool rec;
	const char *name;
	matrix_t *W;
	matrix_t *P;
} db_algorithm_t;

typedef struct {
	// hyperparameters
	db_params_t params;

	// input
	int num_entries;
	image_entry_t *entries;
	int num_labels;
	image_label_t *labels;
	matrix_t *mean_face;

	// algorithms
	db_algorithm_t pca;
	db_algorithm_t lda;
	db_algorithm_t ica;
} database_t;

database_t * db_construct(bool pca, bool lda, bool ica, db_params_t params);
void db_destruct(database_t *db);

void db_train(database_t *db, const char *path);
void db_save(database_t *db, const char *path);
void db_load(database_t *db, const char *path);
void db_recognize(database_t *db, const char *path);

#endif
