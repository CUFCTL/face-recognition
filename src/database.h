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
	int num_classes;
	int num_images;
	int num_dimensions;
	image_entry_t *entries;
	matrix_t *mean_face;

	int pca;
	matrix_t *W_pca;
	matrix_t *P_pca;

	int lda;
	matrix_t *W_lda;
	matrix_t *P_lda;

	int ica;
	matrix_t *W_ica;
	matrix_t *P_ica;
} database_t;

database_t * db_construct(int pca, int lda, int ica);
void db_destruct(database_t *db);

void db_train(database_t *db, const char *path, int n_opt1, int n_opt2);
void db_save(database_t *db, const char *path_tset, const char *path_tdata);

void db_load(database_t *db, const char *path_tset, const char *path_tdata);
void db_recognize(database_t *db, const char *path);

matrix_t * PCA(matrix_t *X, matrix_t **p_D);
matrix_t * LDA(matrix_t *W_pca, matrix_t *X, int c, image_entry_t *entries, int n_opt1, int n_opt2);
matrix_t * ICA(matrix_t *X);

#endif
