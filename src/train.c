/**
 * @file pca_train.c
 *
 * Create a database of eigenfaces from a training set of images.
 *
 * This code implements the following algorithm (needs verification):
 *   m = number of dimensions per image
 *   n = number of images
 *   T = [T_1 ... T_n] (image matrix) (m-by-n)
 *   a = sum(T_i, 1:i:n) / n (mean face) (m-by-1)
 *   A = [(T_1 - a) ... (T_n - a)] (norm. image matrix) (m-by-n)
 *   L = A' * A (surrogate matrix) (n-by-n)
 *   L_ev = eigenvectors of L (n-by-n)
 *   W = A * L_ev (projection matrix / eigenfaces) (m-by-n)
 *   P = W' * A (projected images) (n-by-n)
 */
#include <assert.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lapacke.h>
#include "matrix.h"

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
 * @param path	directory path to scan
 * @param images  pointer to store list of images
 * @return number of images that were found
 */
int get_images(const char* path, char ***images)
{
	// get list of image entries
	struct dirent **entries;
	int num_images = scandir(path, &entries, is_ppm_image, alphasort);

	if ( num_images <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct list of image paths
	*images = (char **)malloc(num_images * sizeof(char *));

	int i;
	for ( i = 0; i < num_images; i++ ) {
		(*images)[i] = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf((*images)[i], "%s/%s", path, entries[i]->d_name);
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
 * @param images	  pointer to array of image paths
 * @param num_images  number of images
 * @return image matrix
 */
matrix_t * get_image_matrix(char **images, int num_images)
{
	// get the image size from the first image
	char header[4];
	uint64_t image_width, image_height, max_brightness;

	FILE *image = fopen(images[0], "r");
	fscanf(image, "%s %" PRIu64 " %" PRIu64 " %" PRIu64 "", header, &image_height, &image_width, &max_brightness);
	fclose(image);
	assert(strcmp(header, "P6") == 0 && max_brightness == 255);

	uint64_t num_pixels = image_width * image_height;

	// read each image into a column of the matrix
	matrix_t *T = m_initialize(num_pixels, num_images);
	unsigned char *pixels = (unsigned char *)malloc(3 * num_pixels * sizeof(unsigned char));

	int i;
	for ( i = 0; i < num_images; i++ ) {
		loadPPMtoMatrixCol(images[i], T, i, pixels);
	}

	free(pixels);

	return T;
}

/**
 * Compute the principal components of a training set.
 *
 * Currently, this function returns all of the n computed
 * eigenvectors, where n is the number of training images.
 *
 * @param A  mean-subtracted image matrix
 * @return projection matrix W_pca
 */
matrix_t * get_projection_matrix_pca(matrix_t *A)
{
	// compute the surrogate matrix L = A' * A
	matrix_t *A_tr = m_transpose(A);
	matrix_t *L = m_matrix_multiply(A_tr, A);

	m_free(A_tr);

	// compute eigenvectors for L
	matrix_t *L_eval = m_initialize(L->numRows, 1);
	matrix_t *L_evec = m_initialize(L->numRows, L->numCols);

	m_eigenvalues_eigenvectors(L, L_eval, L_evec);

	// compute eigenfaces W = A * L_evec
	matrix_t *W = m_matrix_multiply(A, L_evec);

	m_free(L);
	m_free(L_eval);
	m_free(L_evec);

	return W;
}

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./pca-train [images-folder]\n");
		return 1;
	}

	const char *TRAINING_SET_PATH = argv[1];
	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";

	int i;

	// get image matrix T
	char **images;
	int num_images = get_images(TRAINING_SET_PATH, &images);

	matrix_t *A = get_image_matrix(images, num_images);

	// compute the mean face a
	matrix_t *a = m_mean_column(A);

	// save the mean image (for fun and verification)
	// writePPMgrayscale("mean_image.ppm", a, 0, image_height, image_width);

	// compute mean-subtracted image matrix A
	m_normalize_columns(A, a);

	// compute projection matrix W
	matrix_t *W = get_projection_matrix_pca(A);

	// compute projected images P = W' * A
	matrix_t *W_tr = m_transpose(W);

	m_free(W);

	matrix_t *P = m_matrix_multiply(W_tr, A);

	m_free(A);

	// save the image filenames
	FILE *db_training_set = fopen(DB_TRAINING_SET, "w");

	for ( i = 0; i < num_images; i++ ) {
		fprintf(db_training_set, "%s\n", images[i]);
		free(images[i]);
	}
	free(images);

	fclose(db_training_set);

	// save the projected images, transposed eigenfaces, and mean face
	FILE *db_training_data = fopen(DB_TRAINING_DATA, "w");

	m_fwrite(db_training_data, P);
	m_fwrite(db_training_data, W_tr);
	m_fwrite(db_training_data, a);

	m_free(P);
	m_free(W_tr);
	m_free(a);

	fclose(db_training_data);

	return 0;
}
