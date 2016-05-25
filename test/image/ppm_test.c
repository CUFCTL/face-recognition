/**
 * @file ppm_test.c
 *
 * Test suite for the image library.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./ppm-test [image-file]\n");
		return 1;
	}

	const char *IMAGE_FILENAME = argv[1];
	char header[4];
	int image_height, image_width, max_brightness;

	// read the image meta-data
	FILE *image_file = fopen(IMAGE_FILENAME, "r");
	fscanf(image_file, "%s %d %d %d", header, &image_height, &image_width, &max_brightness);
	fclose(image_file);
	assert(strcmp(header, "P6") == 0 && max_brightness == 255);

	// read the image into a column vector
	unsigned char *pixels = (unsigned char *)malloc(3 * image_width * image_height * sizeof(unsigned char));
	matrix_t *image_matrix = m_initialize(UNDEFINED, image_width * image_height, 1);

	loadPPMtoMatrixCol(IMAGE_FILENAME, image_matrix, 0, pixels);

	// write the matrix to a new image file
	writePPMgrayscale(".//wahaha.ppm", image_matrix, 0, image_height, image_width);

	m_free(image_matrix);
	free(pixels);

	return 0;
}
