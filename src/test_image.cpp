/**
 * @file test_image.cpp
 *
 * Test suite for the image library.
 */
#include <stdio.h>
#include "image.h"
#include "logger.h"
#include "matrix.h"

logger_level_t LOGLEVEL = LL_INFO;

int main(int argc, char **argv)
{
	if ( argc != 3 ) {
		fprintf(stderr, "usage: ./test-image [infile] [outfile]\n");
		return 1;
	}

	const char *FILENAME_IN = argv[1];
	const char *FILENAME_OUT = argv[2];

	// map an image to a column vector
	image_t *image = image_construct();
	image_read(image, FILENAME_IN);

	matrix_t *x = m_initialize("x", image->channels * image->width * image->height, 1);
	m_image_read(x, 0, image);

	// map a column vector to an image
	m_image_write(x, 0, image);
	image_write(image, FILENAME_OUT);

	image_destruct(image);
	m_free(x);

	return 0;
}
