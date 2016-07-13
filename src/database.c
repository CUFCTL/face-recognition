/**
 * @file database.c
 *
 * Implementation of the face database.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"
#include "image.h"

/**
 * Get whether an entry is a PGM or PPM image based
 * on the file extension.
 *
 * @param entry
 * @return 1 if entry is PGM or PPM image, 0 otherwise
 */
int is_valid_image(const struct dirent *entry)
{
	return strstr(entry->d_name, ".pgm") != NULL
		|| strstr(entry->d_name, ".ppm") != NULL;
}

/**
 * Get whether an entry is a directory, excluding "." and "..".
 *
 * @param entry
 * @return 1 if entry is a directory, 0 otherwise
 */
int is_directory(const struct dirent *entry)
{
	return entry->d_type == DT_DIR
		&& strcmp(entry->d_name, ".") != 0
		&& strcmp(entry->d_name, "..") != 0;
}

/**
 * Get a list of images in a directory.
 *
 * @param path         directory to scan
 * @param image_names  pointer to store list of images
 * @return number of images that were found
 */
int get_image_names(const char *path, char ***image_names)
{
	// get list of image entries
	struct dirent **entries;
	int num_images = scandir(path, &entries, is_valid_image, alphasort);

	if ( num_images <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct list of image paths
	*image_names = (char **)malloc(num_images * sizeof(char *));

	int i;
	for ( i = 0; i < num_images; i++ ) {
		(*image_names)[i] = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf((*image_names)[i], "%s/%s", path, entries[i]->d_name);
	}

	// clean up
	for ( i = 0; i < num_images; i++ ) {
		free(entries[i]);
	}
	free(entries);

	return num_images;
}

/**
 * Get a list of image entries in a directory.
 *
 * The directory should contain a subdirectory for each
 * class, and each subdirectory should contain the images
 * of its class.
 *
 * @param path           directory path
 * @param image_entries  pointer to store image entries
 * @param num_classes    pointer to store number of classes
 * @return number of images that were found
 */
int get_image_entries(const char *path, database_entry_t **image_entries, int *num_classes)
{
	// get list of classes
	struct dirent **entries;
	*num_classes = scandir(path, &entries, is_directory, alphasort);

	if ( (*num_classes) <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// get list of image names for each class
	char ***classes = (char ***)malloc((*num_classes) * sizeof(char **));
	int *class_sizes = (int *)malloc((*num_classes) * sizeof(int *));

	int num_images = 0;

	int i;
	for ( i = 0; i < (*num_classes); i++ ) {
		char *class_path = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf(class_path, "%s/%s", path, entries[i]->d_name);

		class_sizes[i] = get_image_names(class_path, &classes[i]);
		num_images += class_sizes[i];

		free(class_path);
	}

	// flatten image name lists into a list of entries
	*image_entries = (database_entry_t *)malloc(num_images * sizeof(database_entry_t));

	int num, j;
	for ( num = 0, i = 0; i < (*num_classes); i++ ) {
		for ( j = 0; j < class_sizes[i]; j++ ) {
			(*image_entries)[num] = (database_entry_t) {
				.class = i,
				.name = classes[i][j]
			};

			num++;
		}
	}

	// clean up
	for ( i = 0; i < (*num_classes); i++ ) {
		free(entries[i]);
		free(classes[i]);
	}
	free(entries);
	free(classes);
	free(class_sizes);

	return num_images;
}

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
matrix_t * get_image_matrix(database_entry_t *entries, int num_images)
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
 * @param lda
 * @param ica
 * @return pointer to new database
 */
database_t * db_construct(int lda, int ica)
{
	database_t *db = (database_t *)calloc(1, sizeof(database_t));
	db->lda = lda;
	db->ica = ica;

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
	m_free(db->W_pca_tr);
	m_free(db->P_pca);

	if ( db->lda ) {
		m_free(db->W_lda_tr);
		m_free(db->P_lda);
	}

	if ( db->ica ) {
		m_free(db->W_ica_tr);
		m_free(db->P_ica);
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
	db->num_images = get_image_entries(path, &db->entries, &db->num_classes);

	// compute mean-subtracted image matrix X
	matrix_t *X = get_image_matrix(db->entries, db->num_images);

	db->num_dimensions = X->rows;
	db->mean_face = m_mean_column(X);

	m_subtract_columns(X, db->mean_face);

	// compute PCA representation
	printf("Computing PCA representation...\n");

	db->W_pca_tr = PCA(X);
	db->P_pca = m_product(db->W_pca_tr, X);

	// compute LDA representation
	if ( db->lda ) {
		printf("Computing LDA representation...\n");

		db->W_lda_tr = LDA(db->W_pca_tr, db->P_pca, db->num_classes, db->entries);
		db->P_lda = m_product(db->W_lda_tr, X);
	}

	// compute ICA2 representation
	if ( db->ica ) {
		printf("Computing ICA2 representation...\n");

		db->W_ica_tr = ICA2(db->W_pca_tr, db->P_pca);
		db->P_ica = m_product(db->W_ica_tr, X);
	}

	m_free(X);
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
		fprintf(tset, "%d %s\n", db->entries[i].class, db->entries[i].name);
	}
	fclose(tset);

	// save the mean face and PCA/LDA/ICA representations
	FILE *tdata = fopen(path_tdata, "w");

	m_fwrite(tdata, db->mean_face);
	m_fwrite(tdata, db->W_pca_tr);
	m_fwrite(tdata, db->P_pca);

	if ( db->lda ) {
		m_fwrite(tdata, db->W_lda_tr);
		m_fwrite(tdata, db->P_lda);
	}

	if ( db->ica ) {
		m_fwrite(tdata, db->W_ica_tr);
		m_fwrite(tdata, db->P_ica);
	}

	fclose(tdata);
}

/**
 * Load a database from the file system.
 *
 * @param db  pointer to database
 * @param path_tset   path to read image filenames
 * @param path_tdata  path to read matrix data
 */
void db_load(database_t *db, const char *path_tset, const char *path_tdata)
{
	// get mean face, PDA output, and LDA output
	FILE *tdata = fopen(path_tdata, "r");

	db->mean_face = m_fread(tdata);
	db->W_pca_tr = m_fread(tdata);
	db->P_pca = m_fread(tdata);

	if ( db->lda ) {
		db->W_lda_tr = m_fread(tdata);
		db->P_lda = m_fread(tdata);
	}

	if ( db->ica ) {
		db->W_ica_tr = m_fread(tdata);
		db->P_ica = m_fread(tdata);
	}

	db->num_images = db->P_pca->cols;
	db->num_dimensions = db->mean_face->rows;

	fclose(tdata);

	// get image filenames
	FILE *tset = fopen(path_tset, "r");

	db->entries = (database_entry_t *)malloc(db->num_images * sizeof(database_entry_t));

	int i;
	for ( i = 0; i < db->num_images; i++ ) {
		db->entries[i].name = (char *)malloc(64 * sizeof(char));
		fscanf(tset, "%d %s", &db->entries[i].class, db->entries[i].name);
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
	// get test images
	char **image_names;
	int num_test_images = get_image_names(path, &image_names);
	int index_pca, index_lda, index_ica;
	matrix_t *P_test_pca, *P_test_lda, *P_test_ica;

	// test each image against the database
	image_t *image = image_construct();
	matrix_t *T_i = m_initialize(db->num_dimensions, 1);

	int i;
	for ( i = 0; i < num_test_images; i++ ) {
		// read the test image T_i
		image_read(image, image_names[i]);
		m_image_read(T_i, 0, image);
		m_subtract(T_i, db->mean_face);

		// find the nearest neighbor of P_test for PCA
		P_test_pca = m_product(db->W_pca_tr, T_i);
		index_pca = nearest_neighbor(db->P_pca, P_test_pca, m_dist_L2);

		m_free(P_test_pca);

		// find the nearest neighbor of P_test for LCA
		if ( db->lda ) {
			P_test_lda = m_product(db->W_lda_tr, T_i);
			index_lda = nearest_neighbor(db->P_lda, P_test_lda, m_dist_L2);

			m_free(P_test_lda);
		}

		// find the nearest neighbor of P_test for ICA2
		if ( db->ica ) {
			P_test_ica = m_product(db->W_ica_tr, T_i);
			index_ica = nearest_neighbor(db->P_ica, P_test_ica, m_dist_COS);

			m_free(P_test_ica);
		}

		// print results
		printf("test image: \'%s\'\n", image_names[i]);
		printf("\tPCA:  (class %d) \'%s\'\n", db->entries[index_pca].class, db->entries[index_pca].name);
		if ( db->lda ) printf("\tLDA:  (class %d) \'%s\'\n", db->entries[index_lda].class, db->entries[index_lda].name);
		if ( db->ica ) printf("\tICA2: (class %d) \'%s\'\n", db->entries[index_ica].class, db->entries[index_ica].name);
		putchar('\n');
	}

	// cleanup
	image_destruct(image);
	m_free(T_i);

	for ( i = 0; i < num_test_images; i++ ) {
		free(image_names[i]);
	}
	free(image_names);
}
