/**
 * @file pca_recognize.c
 *
 * Test a set of images against a training set.
 *
 * TODO: Things to cuda-ize (technical term)
 * - matrix multiply
 * - parts of reading in the PPM
 * - vector subtract
 * - distance formula
 * - finding the min
 * - any other loops
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./pca_recognize [images-folder]\n");
		return 1;
	}

	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";
	const char *TEST_SET_PATH = argv[1];

	// get projected images matrix, eigenfaces, and mean face
	FILE *db_training_data = fopen(DB_TRAINING_DATA, "r");

	matrix_t *projected_images = m_fread(db_training_data);
	matrix_t *transposed_eigenfaces = m_fread(db_training_data);
	matrix_t *mean_face = m_fread(db_training_data);

	fclose(db_training_data);

	// get training set images
	FILE *db_training_set = fopen(DB_TRAINING_SET, "r");

	char **training_images = (char **)malloc(projected_images->numCols * sizeof(char *));

	int i;
	for ( i = 0; i < projected_images->numCols; i++ ) {
		training_images[i] = (char *)malloc(64 * sizeof(char));
		fscanf(db_training_set, "%s", training_images[i]);
	}

	fclose(db_training_set);

	// get test set images
	char **test_images;
	int num_test_images = get_images(TEST_SET_PATH, &test_images);

	// test each image against the training set
	unsigned char *pixels = (unsigned char *)malloc(3 * mean_face->numRows * sizeof(unsigned char));

	for ( i = 0; i < num_test_images; i++ ) {
		matrix_t *test_image = m_initialize(UNDEFINED, mean_face->numRows, 1);

		loadPPMtoMatrixCol(test_images[i], test_image, 0, pixels);

		// normalize the test image
		m_subtractColumn(test_image, 0, mean_face);

		// project the test image into the face space
        matrix_t *proj_test_image = m_matrix_multiply(transposed_eigenfaces, test_image);

		// find the training image with the minimum Euclidean distance from the test image
		int min_index = -1;
		double min_dist = -1;

		int j;
		for ( j = 0; j < projected_images->numCols; j++ ) {
			// compute the Euclidean distance between the two images
			// TODO: could use m_norm()
			double dist = 0;

			int k;
			for ( k = 0; k < projected_images->numRows; k++ ) {
				double diff = elem(proj_test_image, k, 0) - elem(projected_images, k, j);
				dist += diff * diff;
			}

			// update the running minimum
			if ( min_dist == -1 || dist < min_dist ) {
				min_index = j;
				min_dist = dist;
			}
		}

		// TODO: consider checking min_dist against a threshold

		printf("test image \'%s\' -> \'%s\'\n", test_images[i], training_images[min_index]);

		m_free(proj_test_image);
		m_free(test_image);
	}

	free(pixels);

	for ( i = 0; i < projected_images->numCols; i++ ) {
		free(training_images[i]);
	}
	free(training_images);

	for ( i = 0; i < num_test_images; i++ ) {
		free(test_images[i]);
	}
	free(test_images);

	m_free(projected_images);
	m_free(transposed_eigenfaces);
	m_free(mean_face);

	return 0;
}
