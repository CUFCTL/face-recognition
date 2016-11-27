/**
 * @file database.c
 *
 * Implementation of the database type.
 */
#include <stdio.h>
#include <stdlib.h>
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
 * @param entries     pointer to list of image entries
 * @param num_images  number of images
 * @return pointer to image matrix
 */
matrix_t * get_image_matrix(image_entry_t *entries, int num_images)
{
	// get the image size from the first image
	image_t *image = image_construct();

	image_read(image, entries[0].name);

	matrix_t *T = m_initialize(image->channels * image->height * image->width, num_images);

	// map each image to a column vector
	m_image_read(T, 0, image);

	int i;
	for ( i = 1; i < num_images; i++ ) {
		image_read(image, entries[i].name);
		m_image_read(T, i, image);
	}

	image_destruct(image);

	return T;
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
		m_dist_L2,
		0, 0
	};
	db->lda = (db_algorithm_t) {
		lda, lda,
		"LDA",
		NULL, NULL,
		m_dist_L2,
		0, 0
	};
	db->ica = (db_algorithm_t) {
		ica, ica,
		"ICA",
		NULL, NULL,
		m_dist_COS,
		0, 0
	};

	if ( LOGLEVEL >= LL_VERBOSE ) {
		printf("Hyperparameters\n");
		printf("PCA\n");
		printf("  pca_n1   %10d\n", db->params.pca_n1);
		printf("LDA\n");
		printf("  lda_n1   %10d\n", db->params.lda_n1);
		printf("  lda_n2   %10d\n", db->params.lda_n2);
		printf("ICA\n");
		printf("  ica_mi   %10d\n", db->params.ica_max_iterations);
		printf("  ica_eps  %10f\n", db->params.ica_epsilon);
		putchar('\n');
	}

	return db;
}

/**
 * Destruct a database.
 *
 * @param db  pointer to database
 */
void db_destruct(database_t *db)
{
	int i;
	for ( i = 0; i < db->num_images; i++ ) {
		free(db->entries[i].name);
	}
	free(db->entries);

	m_free(db->mean_face);

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
 * Train a database with a set of images.
 *
 * @param db	pointer to database
 * @param path  directory of training images
 */
void db_train(database_t *db, const char *path)
{
	timer_push("Training");

	db->num_images = get_directory_rec(path, &db->entries, &db->num_classes);

	// subtract mean from X
	matrix_t *X = get_image_matrix(db->entries, db->num_images);

	db->num_dimensions = X->rows;
	db->mean_face = m_mean_column(X);

	m_subtract_columns(X, db->mean_face);

	// compute PCA representation
	matrix_t *D;

	if ( db->pca.train ) {
		db->pca.W = PCA_cols(X, db->params.pca_n1, &D);
		db->pca.P = m_product(db->pca.W, X, true, false);
	}

	// compute LDA representation
	if ( db->lda.train ) {
		db->lda.W = LDA(db->pca.W, X, db->num_classes, db->entries, db->params.lda_n1, db->params.lda_n2);
		db->lda.P = m_product(db->lda.W, X, true, false);
	}

	// compute ICA representation
	if ( db->ica.train ) {
		db->ica.W = ICA(X, db->params.ica_max_iterations, db->params.ica_epsilon); // W_pca, D
		db->ica.P = m_product(db->ica.W, X, true, false);
	}

	timer_pop();

	// cleanup
	m_free(X);
	m_free(D);
}

/**
 * Save a database to the file system.
 *
 * @param db		  pointer to database
 * @param path_tset   path to save image filenames
 * @param path_tdata  path to save matrix data
 */
void db_save(database_t *db, const char *path_tset, const char *path_tdata)
{
	// save the image filenames
	FILE *tset = fopen(path_tset, "w");

	int i;
	for ( i = 0; i < db->num_images; i++ ) {
		fprintf(tset, "%d %s\n", db->entries[i].ent_class, db->entries[i].name);
	}
	fclose(tset);

	// save the number of images, mean face, PCA/LDA/ICA representations
	FILE *tdata = fopen(path_tdata, "w");

	fwrite(&db->num_images, sizeof(db->num_images), 1, tdata);
	m_fwrite(tdata, db->mean_face);

	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->train ) {
			m_fwrite(tdata, algo->W);
			m_fwrite(tdata, algo->P);
		}
	}

	fclose(tdata);
}

/**
 * Load a database from the file system.
 *
 * @param db          pointer to database
 * @param path_tset   path to read image filenames
 * @param path_tdata  path to read matrix data
 */
void db_load(database_t *db, const char *path_tset, const char *path_tdata)
{
	// read the number of images, mean face, PCA/LDA/ICA representations
	FILE *tdata = fopen(path_tdata, "r");

	fread(&db->num_images, sizeof(db->num_images), 1, tdata);

	db->mean_face = m_fread(tdata);
	db->num_dimensions = db->mean_face->rows;

	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	int i;
	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->train ) {
			algo->W = m_fread(tdata);
			algo->P = m_fread(tdata);
		}
	}

	fclose(tdata);

	// get image filenames
	FILE *tset = fopen(path_tset, "r");

	db->entries = (image_entry_t *)malloc(db->num_images * sizeof(image_entry_t));

	for ( i = 0; i < db->num_images; i++ ) {
		db->entries[i].name = (char *)malloc(64 * sizeof(char));
		fscanf(tset, "%d %s", &db->entries[i].ent_class, db->entries[i].name);
	}

	fclose(tset);
}

/**
 * Find the column vector in a matrix P with minimum distance from
 * a test vector P_test.
 *
 * @param P          pointer to matrix
 * @param P_test     pointer to column vector
 * @param dist_func  pointer to distance function
 * @return index of matching column in P
 */
int nearest_neighbor(matrix_t *P, matrix_t *P_test, dist_func_t dist_func)
{
	int min_index = -1;
	precision_t min_dist = -1;

	int j;
	for ( j = 0; j < P->cols; j++ ) {
		// compute the distance between the two images
		precision_t dist = dist_func(P_test, 0, P, j);

		// update the running minimum
		if ( min_dist == -1 || dist < min_dist ) {
			min_index = j;
			min_dist = dist;
		}
	}

	return min_index;
}

/**
 * Test a set of images against a database.
 *
 * @param db    pointer to database
 * @param path  directory of test images
 */
void db_recognize(database_t *db, const char *path)
{
	timer_push("Recognition");

	// initialize parameters for each recognition algorithm
	db_algorithm_t *algorithms[] = { &db->pca, &db->lda, &db->ica };
	int num_algorithms = sizeof(algorithms) / sizeof(db_algorithm_t *);

	// get test images
	char **image_names;
	int num_test_images = get_directory(path, &image_names);

	// test each image against the database
	image_t *image = image_construct();
	matrix_t *T_i = m_initialize(db->num_dimensions, 1);

	int i;
	for ( i = 0; i < num_test_images; i++ ) {
		// read the test image T_i
		image_read(image, image_names[i]);
		m_image_read(T_i, 0, image);
		m_subtract(T_i, db->mean_face);

		// perform recognition for each algorithm
		int j;
		for ( j = 0; j < num_algorithms; j++ ) {
			db_algorithm_t *algo = algorithms[j];

			if ( algo->rec ) {
				matrix_t *P_test = m_product(algo->W, T_i, true, false);
				algo->rec_index = nearest_neighbor(algo->P, P_test, algo->dist_func);

				m_free(P_test);
			}
		}

		// print results
		if ( LOGLEVEL >= LL_VERBOSE ) {
			printf("test image: \'%s\'\n", rem_base_dir(image_names[i]));
		}

		for ( j = 0; j < num_algorithms; j++ ) {
			db_algorithm_t *algo = algorithms[j];

			if ( algo->rec ) {
				char *rec_name = db->entries[algo->rec_index].name;

				if ( LOGLEVEL >= LL_VERBOSE ) {
					printf("       %s: \'%s\'\n", algo->name, rem_base_dir(rec_name));
				}

				if ( is_same_class(rec_name, image_names[i]) ) {
					algo->num_correct++;
				}
			}
		}

		if ( LOGLEVEL >= LL_VERBOSE ) {
			putchar('\n');
		}
	}

	// print accuracy results
	for ( i = 0; i < num_algorithms; i++ ) {
		db_algorithm_t *algo = algorithms[i];

		if ( algo->rec ) {
			float accuracy = 100.0f * algo->num_correct / num_test_images;

			if ( LOGLEVEL >= LL_VERBOSE ) {
				printf("%s: %d / %d matched, %.2f%%\n", algo->name, algo->num_correct, num_test_images, accuracy);
			}
			else {
				printf("%.2f\n", accuracy);
			}
		}
	}

	timer_pop();

	// cleanup
	image_destruct(image);
	m_free(T_i);

	for ( i = 0; i < num_test_images; i++ ) {
		free(image_names[i]);
	}
	free(image_names);
}
