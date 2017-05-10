/**
 * @file test_image.cpp
 *
 * Test suite for the image library.
 */
#include <stdio.h>
#include "image.h"
#include "matrix.h"

int main(int argc, char **argv)
{
	if ( argc != 3 ) {
		fprintf(stderr, "usage: ./test-image [infile] [outfile]\n");
		return 1;
	}

	const char *FILENAME_IN = argv[1];
	const char *FILENAME_OUT = argv[2];

	// map an image to a column vector
	Image image;
	image.load(FILENAME_IN);

	matrix_t *x = m_initialize("x", image.channels * image.width * image.height, 1);
	m_image_read(x, 0, image);

	// map a column vector to an image
	m_image_write(x, 0, image);
	image.save(FILENAME_OUT);

	m_free(x);

	return 0;
}
