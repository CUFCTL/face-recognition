/**
 * @file database.c
 *
 * Implementation of the face database.
 */
#include <assert.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "database.h"

/**
 * Get whether a file is a PPM image based on the
 * file extension.
 *
 * @param entry
 * @return 1 if file is PPM image, 0 otherwise
 */
int is_ppm_image(const struct dirent *entry)
{
	return strstr(entry->d_name, ".ppm") != NULL;
}

/**
 * Get a list of images in a directory.
 *
 * @param path         directory to scan
 * @param image_names  pointer to store list of images
 * @return number of images that were found
 */
int get_image_names(const char* path, char ***image_names)
{
	// get list of image entries
	struct dirent **entries;
	int num_images = scandir(path, &entries, is_ppm_image, alphasort);

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
 * Load a collection of PPM images into a matrix.
 *
 * The image matrix has size m x n, where m is the number of
 * pixels in each image and n is the number of images. The
 * images in the training set must all have the same size.
 *
 * @param image_names  pointer to list of image names
 * @param num_images   number of images
 * @return image matrix
 */
matrix_t * get_image_matrix(char **image_names, int num_images)
{
	// get the image size from the first image
	char header[4];
	uint64_t image_width, image_height, max_brightness;

	FILE *image = fopen(image_names[0], "r");
	fscanf(image, "%s %" PRIu64 " %" PRIu64 " %" PRIu64 "", header, &image_height, &image_width, &max_brightness);
	fclose(image);
	assert(strcmp(header, "P6") == 0 && max_brightness == 255);

	uint64_t num_pixels = image_width * image_height;

	// read each image into a column of the matrix
	matrix_t *T = m_initialize(num_pixels, num_images);
	unsigned char *pixels = (unsigned char *)malloc(3 * num_pixels * sizeof(unsigned char));

	int i;
	for ( i = 0; i < num_images; i++ ) {
		loadPPMtoMatrixCol(image_names[i], T, i, pixels);
	}

	free(pixels);

	return T;
}

/**
 * Construct a database.
 *
 * @return pointer to new database
 */
database_t * db_construct()
{
    database_t *db = (database_t *)malloc(sizeof(database_t));

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
		free(db->image_names[i]);
	}
	free(db->image_names);

    m_free(db->mean_face);
	m_free(db->W_pca_tr);
//	m_free(db->W_lda_tr);
//	m_free(db->W_ica_tr);
    m_free(db->images_proj);

    free(db);
}

/**
 * Train a database with a set of images.
 *
 * @param db    pointer to database
 * @param path  directory of training images
 */
void db_train(database_t *db, const char *path)
{
    db->num_images = get_image_names(path, &db->image_names);

    // compute mean-subtracted image matrix A
    matrix_t *A = get_image_matrix(db->image_names, db->num_images);

    db->num_dimensions = A->numRows;
    db->mean_face = m_mean_column(A);

	m_normalize_columns(A, db->mean_face);

	// compute projection matrix W_pca'
	matrix_t *W_pca = get_projection_matrix_PCA(A);
//	matrix_t *W_lda = get_projection_matrix_LDA(A);
//	matrix_t *W_ica = get_projection_matrix_ICA(A);

	db->W_pca_tr = m_transpose(W_pca);

	m_free(W_pca);

	// compute projected images P = W_pca' * A
	db->images_proj = m_matrix_multiply(db->W_pca_tr, A);

    m_free(A);
}

/**
 * Save a database to the file system.
 *
 * @param db          pointer to database
 * @param path_tset   path to save image filenames
 * @param path_tdata  path to save matrix data
 */
void db_save(database_t *db, const char *path_tset, const char *path_tdata)
{
	// save the image filenames
	FILE *tset = fopen(path_tset, "w");

	int i;
	for ( i = 0; i < db->num_images; i++ ) {
		fprintf(tset, "%s\n", db->image_names[i]);
	}
	fclose(tset);

	// save the mean face, projection matrix, and projected images
	FILE *tdata = fopen(path_tdata, "w");

	m_fwrite(tdata, db->mean_face);
	m_fwrite(tdata, db->W_pca_tr);
	m_fwrite(tdata, db->images_proj);
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
	// get mean face, projection matrix, and projected images
	FILE *db_training_data = fopen(path_tdata, "r");

	db->mean_face = m_fread(db_training_data);
	db->W_pca_tr = m_fread(db_training_data);
	db->images_proj = m_fread(db_training_data);

    db->num_images = db->images_proj->numCols;
    db->num_dimensions = db->mean_face->numRows;

	fclose(db_training_data);

	// get image filenames
	FILE *db_training_set = fopen(path_tset, "r");

	db->image_names = (char **)malloc(db->num_images * sizeof(char *));

	int i;
	for ( i = 0; i < db->num_images; i++ ) {
		db->image_names[i] = (char *)malloc(64 * sizeof(char));
		fscanf(db_training_set, "%s", db->image_names[i]);
	}

	fclose(db_training_set);
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

	// test each image against the database
	unsigned char *pixels = (unsigned char *)malloc(3 * db->num_dimensions * sizeof(unsigned char));

    int i;
	for ( i = 0; i < num_test_images; i++ ) {
		// compute the mean-subtracted test image
		matrix_t *T_i = m_initialize(db->num_dimensions, 1);

		loadPPMtoMatrixCol(image_names[i], T_i, 0, pixels);
		m_normalize_columns(T_i, db->mean_face);

		// compute the projected test image T_i_proj = W' * T_i
		matrix_t *T_i_proj = m_matrix_multiply(db->W_pca_tr, T_i);

		// find the training image with the minimum Euclidean distance from the test image
		int min_index = -1;
		double min_dist = -1;

		int j;
		for ( j = 0; j < db->num_images; j++ ) {
			// compute the Euclidean distance between the two images
			double dist = 0;

			int k;
			for ( k = 0; k < T_i_proj->numRows; k++ ) {
				double diff = elem(T_i_proj, k, 0) - elem(db->images_proj, k, j);
				dist += diff * diff;
			}

			// update the running minimum
			if ( min_dist == -1 || dist < min_dist ) {
				min_index = j;
				min_dist = dist;
			}
		}

		printf("test image \'%s\' -> \'%s\'\n", image_names[i], db->image_names[min_index]);

		m_free(T_i_proj);
		m_free(T_i);
	}

	free(pixels);

	for ( i = 0; i < num_test_images; i++ ) {
		free(image_names[i]);
	}
	free(image_names);
}
