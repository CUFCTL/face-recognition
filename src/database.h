/**
 * @file database.h
 *
 * Interface definitions for a face database.
 */

#ifndef DATABASE_H
#define DATABASE_H

#include "matrix.h"

typedef precision_t (*dist_func_t)(matrix_t *, int, matrix_t *, int);

typedef struct {
	int class;
	char *name;
} database_entry_t;

typedef struct {
	int num_classes;
	int num_images;
	int num_dimensions;
	database_entry_t *entries;
	matrix_t *mean_face;
	matrix_t *W_pca_tr;
	matrix_t *W_lda_tr;
	matrix_t *W_ica_tr;
	matrix_t *P_pca;
	matrix_t *P_lda;
	matrix_t *P_ica;
} database_t;

typedef struct {
	int pca;
	int lda;
	int ica;
} args_t;

database_t * db_construct();
void db_destruct(database_t *db, args_t * args);

void db_train(database_t *db, const char *path, args_t * args);
void db_save(database_t *db, const char *path_tset, const char *path_tdata, args_t * args);

void db_load(database_t *db, const char *path_tset, const char *path_tdata, args_t * args);
void db_recognize(database_t *db, const char *path, args_t * args);

matrix_t * PCA(matrix_t *X);
matrix_t * LDA(matrix_t *W_pca_tr, matrix_t *P_pca, int c, database_entry_t *entries);
matrix_t * ICA2(matrix_t *W_pca_tr, matrix_t *P_pca);

#endif
