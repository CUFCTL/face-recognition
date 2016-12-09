/**
 * @file image.c
 *
 * Implementation of the image type.
 *
 * The following formats are supported:
 * - binary PGM (P5)
 * - binary PPM (P6)
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image.h"

/**
 * Construct a PPM image.
 *
 * @return pointer to empty PPM image
 */
image_t * image_construct()
{
	image_t *image = (image_t *)malloc(sizeof(image_t));
	image->channels = 0;
	image->height = 0;
	image->width = 0;
	image->max_value = 0;
	image->pixels = NULL;

	return image;
}

/**
 * Destruct an image.
 *
 * @param image
 */
void image_destruct(image_t *image)
{
	free(image->pixels);
	free(image);
}

/**
 * Helper function to skip comments in a PGM/PPM image.
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
 * Read an image from a PGM/PPM file.
 *
 * @param image
 * @param path
 */
void image_read(image_t *image, const char *path)
{
	FILE *in = fopen(path, "r");

	char header[4];

	fscanf(in, "%s", header);
	if ( strcmp(header, "P5") == 0 ) {
		image->channels = 1;
	}
	else if ( strcmp(header, "P6") == 0 ) {
		image->channels = 3;
	}
	else {
		fprintf(stderr, "error: cannot read image \'%s\'\n", path);
		exit(1);
	}

	skip_to_next_value(in);

	fscanf(in, "%d", &image->height);
	skip_to_next_value(in);

	fscanf(in, "%d", &image->width);
	skip_to_next_value(in);

	fscanf(in, "%d", &image->max_value);
	fgetc(in);

	int num = image->channels * image->height * image->width;

	if ( image->pixels == NULL ) {
		image->pixels = (unsigned char *)malloc(num * sizeof(unsigned char));
	}

	fread(image->pixels, sizeof(unsigned char), num, in);

	fclose(in);
}

/**
 * Write an image to a PGM/PPM file.
 *
 * @param image
 * @param path
 */
void image_write(image_t *image, const char *path)
{
	FILE *out = fopen(path, "w");

	// write image header
	if ( image->channels == 1 ) {
		fprintf(out, "P5");
	}
	else if ( image->channels == 3 ) {
		fprintf(out, "P6");
	}
	else {
		fprintf(stderr, "error: cannot write image \'%s\'\n", path);
		exit(1);
	}

	fprintf(out, "%d %d %d\n", image->height, image->width, image->max_value);

	// write pixel data
	fwrite(image->pixels, sizeof(unsigned char), image->channels * image->height * image->width, out);

	fclose(out);
}
