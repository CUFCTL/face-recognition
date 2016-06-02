/**
 * @file ppm.c
 *
 * Implementation of the PPM image type.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ppm.h"

/**
 * Construct a PPM image.
 *
 * @return pointer to empty PPM image
 */
ppm_t * ppm_construct()
{
    ppm_t *image = (ppm_t *)malloc(sizeof(ppm_t));
    image->height = 0;
    image->width = 0;
    image->max_value = 0;
    image->pixels = NULL;

    return image;
}

/**
 * Destruct a PPM image.
 *
 * @param image  pointer to PPM image
 */
void ppm_destruct(ppm_t *image)
{
    free(image->pixels);
    free(image);
}

/**
 * Helper function to skip comments in a PPM image.
 */
void skip_to_next_value(FILE* in)
{
	char c = fgetc(in);
	while ( c == '#' || isspace(c) ) {
		if ( c == '#' ) {
			while ( c != '\n' ) {
				c = fgetc(in);
			}
		}
		else {
			while ( isspace(c) ) {
				c = fgetc(in);
			}
		}
	}

	ungetc(c, in);
}

/**
 * Read a PPM image from a file.
 *
 * @param image  pointer to PPM image
 * @param path   image filename
 */
void ppm_read(ppm_t *image, const char *path)
{
	FILE *in = fopen(path, "r");

	char header[4];

	fscanf(in, "%s", header);
	if ( strcmp(header, "P3") != 0 && strcmp(header, "P6") != 0 ) {
		fprintf(stderr, "error: PPM is not P3 or P6\n");
		exit(1);
	}

	skip_to_next_value(in);
	fscanf(in, "%d", &image->height);

	skip_to_next_value(in);
	fscanf(in, "%d", &image->width);

	skip_to_next_value(in);
	fscanf(in, "%d", &image->max_value);

    if ( image->pixels == NULL ) {
        image->pixels = (unsigned char *)malloc(3 * image->height * image->width * sizeof(unsigned char));
    }

	skip_to_next_value(in);
	fread(image->pixels, sizeof(unsigned char), 3 * image->height * image->width, in);

	fclose(in);
}

/**
 * Write a PPM image to a file.
 *
 * @param image  pointer to PPM image
 * @param path   image filename
 */
void ppm_write(ppm_t *image, const char *path)
{
	FILE *out = fopen(path, "w");

	// write image header
	fprintf(out, "P6 %d %d %d\n", image->height, image->width, image->max_value);

	// write pixel data
	fwrite(image->pixels, sizeof(unsigned char), 3 * image->height * image->width, out);

	fclose(out);
}
