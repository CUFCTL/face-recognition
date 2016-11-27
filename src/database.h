/**
 * @file database.h
 *
 * Interface definitions for the database type.
 */
#ifndef DATABASE_H
#define DATABASE_H

#include "image_entry.h"
#include "matrix.h"

extern int VERBOSE;

typedef precision_t (*dist_func_t)(matrix_t *, int, matrix_t *, int);

typedef struct {
	// PCA hyperparameters
	int pca_n1;

	// PCA hyperparameters
	int lda_n1;
	int lda_n2;

	// ICA hyperparameters
	int ica_max_iterations;
	precision_t ica_epsilon;
} db_params_t;

typedef struct {
	bool train;
	bool rec;
	const char * name;
	matrix_t *W;
	matrix_t *P;
	dist_func_t dist_func;
	int rec_index;
	int num_correct;
} db_algorithm_t;

typedef struct {
	// hyperparameters
	db_params_t params;

	// input
	int num_classes;
	int num_images;
	int num_dimensions;
	image_entry_t *entries;
	matrix_t *mean_face;

	// algorithms
	db_algorithm_t pca;
	db_algorithm_t lda;
	db_algorithm_t ica;
} database_t;

database_t * db_construct(int pca, int lda, int ica, db_params_t params);
void db_destruct(database_t *db);

void db_train(database_t *db, const char *path);
void db_save(database_t *db, const char *path_tset, const char *path_tdata);
void db_load(database_t *db, const char *path_tset, const char *path_tdata);
void db_recognize(database_t *db, const char *path);

matrix_t * PCA_cols(matrix_t *X, int n_opt1, matrix_t **p_D);
matrix_t * PCA_rows(matrix_t *X, matrix_t **p_D);
matrix_t * LDA(matrix_t *W_pca, matrix_t *X, int c, image_entry_t *entries, int n_opt1, int n_opt2);
matrix_t * ICA(matrix_t *X, int max_iterations, precision_t epsilon);

#endif
