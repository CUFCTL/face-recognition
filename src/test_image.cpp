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

	Matrix x("x", image.channels * image.width * image.height, 1);
	x.image_read(0, image);

	// map a column vector to an image
	x.image_write(0, image);
	image.save(FILENAME_OUT);

	return 0;
}
