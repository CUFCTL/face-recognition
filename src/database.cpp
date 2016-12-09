/**
 * @file database.c
 *
 * Implementation of the database type.
 */
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"
#include "image.h"
#include "timer.h"

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
 * Construct a database.
 *
 * @param pca
 * @param lda
 * @param ica
 * @param params
 * @return pointer to new database
 */
database_t * db_construct(int pca, int lda, int ica, db_params_t params)
{
	database_t *db = (database_t *)calloc(1, sizeof(database_t));
	db->params = params;

	db->pca = (db_algorithm_t) {
		pca || lda || ica, pca,
		"PCA",
		NULL, NULL,
		m_dist_L2
	};
	db->lda = (db_algorithm_t) {
		lda, lda,
		"LDA",
		NULL, NULL,
		m_dist_L2
	};
	db->ica = (db_algorithm_t) {
		ica, ica,
		"ICA",
		NULL, NULL,
		m_dist_COS
	};

	if ( LOGGER(LL_VERBOSE) ) {
		int len = 20;

		printf("Hyperparameters\n");
		printf("PCA\n");
		printf("  %-*s  %10d\n", len, "pca_n1", db->params.pca_n1);
		printf("LDA\n");
		printf("  %-*s  %10d\n", len, "lda_n1", db->params.lda_n1);
		printf("  %-*s  %10d\n", len, "lda_n2", db->params.lda_n2);
		printf("ICA\n");
		printf("  %-*s  %10d\n", len, "ica_num_ic", db->params.ica_num_ic);
		printf("  %-*s  %10d\n", len, "ica_max_iterations", db->params.ica_max_iterations);
		printf("  %-*s  %10f\n", len, "ica_epsilon", db->params.ica_epsilon);
		printf("kNN\n");
		printf("  %-*s  %10d\n", len, "knn_k", db->params.knn_k);
		putchar('\n');
	}

	return db;
}

/**
 * Destruct a database.
 *
 * @param db
 */
void db_destruct(database_t *db)
{
	// free entries
	int i;
	for ( i = 0; i < db->num_entries; i++ ) {
		free(db->entries[i].name);
	}
	free(db->entries);

	// free labels
	for ( i = 0; i < db->num_labels; i++ ) {
		free(db->labels[i].name);
	}
	free(db->labels);

	// free mean face
	m_free(db->mean_face);

	// free algorithm data
	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->train ) {
			m_free(algo->W);
			m_free(algo->P);
		}
	}

	free(db);
}

/**
 * Perform training on a training set.
 *
 * @param db
 * @param path
 */
void db_train(database_t *db, const char *path)
{
	timer_push("Training");

	// get entries, labels
	db->num_entries = get_directory(path, &db->entries, &db->num_labels, &db->labels);

	// get image matrix X
	matrix_t *X = get_image_matrix(db->entries, db->num_entries);

	// subtract mean from X
	db->mean_face = m_mean_column("m", X);
	m_subtract_columns(X, db->mean_face);

	// compute PCA representation
	matrix_t *D;

	if ( db->pca.train ) {
		db->pca.W = PCA_cols(X, db->params.pca_n1, &D);
		db->pca.P = m_product("P_pca", db->pca.W, X, true, false);
	}

	// compute LDA representation
	if ( db->lda.train ) {
		db->lda.W = LDA(db->pca.W, X, db->num_labels, db->entries, db->params.lda_n1, db->params.lda_n2);
		db->lda.P = m_product("P_lda", db->lda.W, X, true, false);
	}

	// compute ICA representation
	if ( db->ica.train ) {
		db->ica.W = ICA(X, db->params.ica_num_ic, db->params.ica_max_iterations, db->params.ica_epsilon); // W_pca, D
		db->ica.P = m_product("P_ica", db->ica.W, X, true, false);
	}

	timer_pop();

	// cleanup
	m_free(X);
	m_free(D);
}

/**
 * Save a database to a data file.
 *
 * @param db
 * @param path
 */
void db_save(database_t *db, const char *path)
{
	FILE *file = fopen(path, "w");

	// save labels
	fwrite(&db->num_labels, sizeof(int), 1, file);

	int i;
	for ( i = 0; i < db->num_labels; i++ ) {
		fwrite(&db->labels[i].id, sizeof(int), 1, file);

		int num = strlen(db->labels[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(db->labels[i].name, sizeof(char), num, file);
	}

	// save entries
	fwrite(&db->num_entries, sizeof(int), 1, file);

	for ( i = 0; i < db->num_entries; i++ ) {
		fwrite(&db->entries[i].label->id, sizeof(int), 1, file);

		int num = strlen(db->entries[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(db->entries[i].name, sizeof(char), num, file);
	}

	// save mean face
	m_fwrite(file, db->mean_face);

	// save algorithm data
	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->train ) {
			m_fwrite(file, algo->W);
			m_fwrite(file, algo->P);
		}
	}

	fclose(file);
}

/**
 * Load a database from a file.
 *
 * @param db
 * @param path
 */
void db_load(database_t *db, const char *path)
{
	FILE *file = fopen(path, "r");

	// read labels
	fread(&db->num_labels, sizeof(int), 1, file);

	db->labels = (image_label_t *)malloc(db->num_labels * sizeof(image_label_t));

	int i;
	for ( i = 0; i < db->num_labels; i++ ) {
		fread(&db->labels[i].id, sizeof(int), 1, file);

		int num;
		fread(&num, sizeof(int), 1, file);

		db->labels[i].name = (char *)malloc(num * sizeof(char));
		fread(db->labels[i].name, sizeof(char), num, file);
	}

	// read entries
	fread(&db->num_entries, sizeof(int), 1, file);

	db->entries = (image_entry_t *)malloc(db->num_entries * sizeof(image_entry_t));

	for ( i = 0; i < db->num_entries; i++ ) {
		int label_id;
		fread(&label_id, sizeof(int), 1, file);

		db->entries[i].label = &db->labels[label_id];

		int num;
		fread(&num, sizeof(int), 1, file);

		db->entries[i].name = (char *)malloc(num * sizeof(char));
		fread(db->entries[i].name, sizeof(char), num, file);
	}

	// read mean face
	db->mean_face = m_fread(file);

	// read algorithm data
	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->train ) {
			algo->W = m_fread(file);
			algo->P = m_fread(file);
		}
	}

	fclose(file);
}

typedef struct {
	image_label_t *label;
	precision_t dist;
} neighbor_t;

typedef struct {
	image_label_t *label;
	int count;
} label_count_t;

int kNN_compare(const void *a, const void *b)
{
	neighbor_t *n1 = (neighbor_t *)a;
	neighbor_t *n2 = (neighbor_t *)b;

	return (int)(n1->dist - n2->dist);
}

/**
 * Classify an observation using k-nearest neighbors.
 *
 * @param X
 * @param Y
 * @param X_test
 * @param i
 * @param k
 * @param dist_func
 * @return predicted label of the test observation
 */
image_label_t * kNN(matrix_t *X, image_entry_t *Y, matrix_t *X_test, int i, int k, dist_func_t dist_func)
{
	// compute distance between X_test_i and each observation in X
	neighbor_t *neighbors = (neighbor_t *)malloc(X->cols * sizeof(neighbor_t));

	int j;
	for ( j = 0; j < X->cols; j++ ) {
		neighbors[j].label = Y[j].label;
		neighbors[j].dist = dist_func(X_test, i, X, j);
	}

	// sort the neighbors by distance
	qsort(neighbors, X->cols, sizeof(neighbor_t), kNN_compare);

	// determine the mode of the k nearest labels
	// TODO: maybe replace with mode function
	label_count_t *counts = (label_count_t *)calloc(k, sizeof(label_count_t));

	for ( j = 0; j < k; j++ ) {
		int n = 0;
		while ( counts[n].label != NULL && counts[n].label != neighbors[j].label ) {
			n++;
		}

		counts[n].label = neighbors[j].label;
		counts[n].count++;
	}

	label_count_t *max = NULL;

	for ( j = 0; counts[j].label != NULL; j++ ) {
		if ( max == NULL || max->count < counts[j].count ) {
			max = &counts[j];
		}
	}

	return max->label;
}

/**
 * Perform recognition on a test set.
 *
 * @param db
 * @param path
 */
void db_recognize(database_t *db, const char *path)
{
	timer_push("Recognition");

	// get entries, labels
	image_label_t *labels;
	int num_labels;

	image_entry_t *entries;
	int num_entries = get_directory(path, &entries, &num_labels, &labels);

	// get image matrix X_test
	matrix_t *X_test = get_image_matrix(entries, num_entries);

	// subtract database mean from X_test
	m_subtract_columns(X_test, db->mean_face);

	// initialize list of recognition algorithms
	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	// perform recognition for each algorithm
	int i;
	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->rec ) {
			// compute projected test images
			matrix_t *P_test = m_product("P_test", algo->W, X_test, true, false);

			// compute labels for each test image
			image_label_t **rec_labels = (image_label_t **)malloc(num_entries * sizeof(image_label_t *));

			int j;
			for ( j = 0; j < num_entries; j++ ) {
				rec_labels[j] = kNN(algo->P, db->entries, P_test, j, db->params.knn_k, algo->dist_func);
			}

			// compute accuracy
			int num_correct = 0;

			for ( j = 0; j < num_entries; j++ ) {
				if ( strcmp(rec_labels[j]->name, entries[j].label->name) == 0 ) {
					num_correct++;
				}
			}

			float accuracy = 100.0f * num_correct / num_entries;

			// print results
			if ( LOGGER(LL_VERBOSE) ) {
				printf("  %s\n", algo->name);

				for ( j = 0; j < num_entries; j++ ) {
					const char *s = (strcmp(rec_labels[j]->name, entries[j].label->name) != 0)
						? "(!)"
						: "";

					printf("    %-10s -> %-4s %s\n", basename(entries[j].name), rec_labels[j]->name, s);
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
		}
	}

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
