/**
 * @file database.h
 *
 * Interface definitions for the database type.
 */
#ifndef DATABASE_H
#define DATABASE_H

#include "image_entry.h"
#include "logger.h"
#include "matrix.h"

typedef precision_t (*dist_func_t)(matrix_t *, int, matrix_t *, int);

typedef struct {
	int n1;
	dist_func_t dist;
} pca_params_t;

typedef struct {
	int n1;
	int n2;
	dist_func_t dist;
} lda_params_t;

typedef struct {
	int num_ic;
	int max_iterations;
	precision_t epsilon;
	dist_func_t dist;
} ica_params_t;

typedef struct {
	int k;
} knn_params_t;

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
	dist_func_t dist_func;
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

// feature extraction algorithms
matrix_t * PCA(matrix_t *X, int n1, matrix_t **p_D);
matrix_t * LDA(matrix_t *W_pca, matrix_t *X, int c, image_entry_t *entries, int n1, int n2);
matrix_t * ICA(matrix_t *X, int num_ic, int max_iterations, precision_t epsilon);

// classification algorithms
image_label_t * kNN(matrix_t *X, image_entry_t *Y, matrix_t *X_test, int i, int k, dist_func_t dist_func);

#endif
