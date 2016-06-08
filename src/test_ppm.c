/**
 * @file test_ppm.c
 *
 * Test suite for the image library.
 */
#include <stdio.h>
#include "matrix.h"
#include "ppm.h"

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./test-ppm [image-file]\n");
		return 1;
	}

	const char *FILENAME_IN = argv[1];
	const char *FILENAME_OUT = "wahaha.ppm";

	// map an image to a column vector
	ppm_t *image = ppm_construct();
	ppm_read(image, FILENAME_IN);

	matrix_t *x = m_initialize(image->channels * image->width * image->height, 1);
	m_ppm_read(x, 0, image);

	// map a column vector to an image
	m_ppm_write(x, 0, image);
	ppm_write(image, FILENAME_OUT);

	ppm_destruct(image);
	m_free(x);

	return 0;
}
