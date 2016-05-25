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
 *   E = A * L_ev (eigenfaces) (m-by-n)
 *   P = E' * A (projected images) (n-by-n)
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

	// get image matrix
	char **images;
	int num_images = get_images(TRAINING_SET_PATH, &images);

	matrix_t *T = get_image_matrix(images, num_images);

	// compute the mean face
	matrix_t *a = m_meanRows(T);

	// save the mean image (for fun and verification)
	// writePPMgrayscale("mean_image.ppm", a, 0, image_height, image_width);

	// normalize each face with the mean face
	m_normalize_columns(T, a);

	// compute the surrogate matrix L = A' * A
	matrix_t *A = T;
	matrix_t *A_tr = m_transpose(A);
	matrix_t *L = m_matrix_multiply(A_tr, A);

	m_free(A_tr);

	// compute eigenvectors for L
	// TODO: replace with m_eigenvalues_eigenvectors()
	double w[L->numRows * L->numCols];
	int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', L->numRows, L->data, L->numCols, w);

	if ( info > 0 ) {
		fprintf(stderr, "The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}

	matrix_t *L_ev = L;

	// compute eigenfaces E = A * L_ev
	matrix_t *E = m_matrix_multiply(A, L_ev);

	m_free(L_ev);

	// compute transposed eigenfaces E'
	matrix_t *E_tr = m_transpose(E);

	m_free(E);

	// compute projected images P = E' * A
	matrix_t *P = m_matrix_multiply(E_tr, A);

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
	m_fwrite(db_training_data, E_tr);
	m_fwrite(db_training_data, a);

	m_free(P);
	m_free(E_tr);
	m_free(a);

	fclose(db_training_data);

	return 0;
}
