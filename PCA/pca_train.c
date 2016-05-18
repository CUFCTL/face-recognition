/**
 * @file pca_train.c
 *
 * Create a database of faces from a training set of images.
 */
#include <assert.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <lapacke.h>
#include "matrixOperations.h"

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
 * @param path    directory path to scan
 * @param images  pointer to store list of images
 * @return number of images that were found
 */
uint64_t get_images(const char* path, char ***images)
{
	// get list of image entries
    struct dirent **entries;
    uint64_t num_images = scandir(path, &entries, is_ppm_image, alphasort);

	if ( num_images <= 0 ) {
        perror("scandir");
		exit(1);
    }

    // construct list of image paths
    *images = (char **)malloc(num_images * sizeof(char *));

	uint64_t i;
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
 * @param images      pointer to array of image paths
 * @param num_images  number of images
 * @return image matrix
 */
matrix_t * get_image_matrix(char **images, uint64_t num_images)
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
	matrix_t *T = m_initialize(UNDEFINED, num_pixels, num_images);
	unsigned char *pixels = (unsigned char *)malloc(3 * num_pixels * sizeof(unsigned char));

	uint64_t i;
	for ( i = 0; i < num_images; i++ ) {
		loadPPMtoMatrixCol(images[i], T, i, pixels);
	}

	free(pixels);

    return T;
}

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./pca-train [images-folder]");
		return 1;
	}

	const char *TRAINING_SET_PATH = argv[1];
	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";

	clock_t start, end;
	uint64_t i;

    // get image matrix
    char **images;
    uint64_t num_images = get_images(TRAINING_SET_PATH, &images);

	start = clock();

    matrix_t *T = get_image_matrix(images, num_images);

	end = clock();
	printf("time to load image matrix T: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	// compute the mean face
	start = clock();

	matrix_t *M = m_meanRows(T);

	end = clock();
	printf("time to calc mean face M: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	// save the mean image (for fun and verification)
	// writePPMgrayscale("mean_image.ppm", M, 0, image_height, image_width);

	// normalize each face with the mean face
	matrix_t *A = T;

	start = clock();

	for ( i = 0; i < num_images; i++ ) {
        // TODO: test this function
	    m_subtractColumn(A, i, M);
	}

	end = clock();
	printf("time to calc norm matrix A: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	// compute the surrogate matrix L = A' * A
	start = clock();

    matrix_t *At = m_transpose(A);
	matrix_t *L = m_matrix_multiply(At, A);

	end = clock();
	printf("time to calc surrogate matrix L: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

    m_free(At);

	// compute eigenvectors for L
	start = clock();

    double w[L->numRows * L->numCols];
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', L->numRows, L->data, L->numCols, w);

    if ( info > 0 ) {
        fprintf(stderr, "The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

	end = clock();
	printf("time to calc eigenvectors: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	// m_free(L);
    matrix_t *L_eigenvectors = L;

	// compute eigenfaces = A * L_eigenvectors
	start = clock();

	matrix_t *eigenfaces = m_matrix_multiply(A, L_eigenvectors);

	end = clock();
	printf("time to calc eigenfaces: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	m_free(L_eigenvectors);

	// transpose eigenfaces
	start = clock();

	matrix_t *eigenfaces_transposed = m_transpose(eigenfaces);

	end = clock();
	printf("time to transpose eigenfaces: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

	m_free(eigenfaces);

	// compute projected images = eigenfaces' * A
	start = clock();

	matrix_t *projected_images = m_matrix_multiply(eigenfaces_transposed, A);

	end = clock();
	printf("time to calc projected images: %.3f s\n", (double)(end - start)/CLOCKS_PER_SEC);

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

	m_fwrite(db_training_data, projected_images);
	m_fwrite(db_training_data, eigenfaces_transposed);
	m_fwrite(db_training_data, M);

	m_free(projected_images);
	m_free(eigenfaces_transposed);
	m_free(M);
	fclose(db_training_data);

	return 0;
}
