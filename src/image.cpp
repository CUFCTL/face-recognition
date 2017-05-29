/**
 * @file image.cpp
 *
 * Implementation of the image type.
 *
 * The following formats are supported:
 * - binary PGM (P5)
 * - binary PPM (P6)
 */
#include <ctype.h>
#include <stdio.h>
#include "image.h"

/**
 * Construct an image.
 */
Image::Image()
{
	this->channels = 0;
	this->width = 0;
	this->height = 0;
	this->max_value = 0;
	this->pixels = nullptr;
}

/**
 * Destruct an image.
 */
Image::~Image()
{
	delete[] this->pixels;
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
 * Load an image from a PGM/PPM file.
 *
 * @param path
 */
void Image::load(const std::string& path)
{
	FILE *in = fopen(path.c_str(), "r");

	// read image header
	char buffer[4];
	fscanf(in, "%s", buffer);

	std::string header(buffer);
	int channels;

	if ( header == "P5" ) {
		channels = 1;
	}
	else if ( header == "P6" ) {
		channels = 3;
	}
	else {
		fprintf(stderr, "error: cannot read image \'%s\'\n", path.c_str());
		exit(1);
	}

	skip_to_next_value(in);

	// read image metadata
	int width;
	int height;
	int max_value;

	fscanf(in, "%d", &width);
	skip_to_next_value(in);

	fscanf(in, "%d", &height);
	skip_to_next_value(in);

	fscanf(in, "%d", &max_value);
	fgetc(in);

	// verify that image sizes are equal (if reloading)
	int num1 = channels * width * height;
	int num2 = this->channels * this->width * this->height;

	if ( this->pixels == nullptr ) {
		this->pixels = new unsigned char[num1];
	}
	else if ( num1 != num2 ) {
		fprintf(stderr, "error: unequal sizes on image reload\n");
		exit(1);
	}

	this->channels = channels;
	this->width = width;
	this->height = height;
	this->max_value = max_value;

	// read pixel data
	fread(this->pixels, sizeof(unsigned char), num1, in);

	fclose(in);
}

/**
 * Save an image to a PGM/PPM file.
 *
 * @param path
 */
void Image::save(const std::string& path)
{
	FILE *out = fopen(path.c_str(), "w");

	// write image header
	if ( this->channels == 1 ) {
		fprintf(out, "P5\n");
	}
	else if ( this->channels == 3 ) {
		fprintf(out, "P6\n");
	}
	else {
		fprintf(stderr, "error: cannot write image \'%s\'\n", path.c_str());
		exit(1);
	}

	// write image metadata
	fprintf(out, "%d %d %d\n", this->width, this->height, this->max_value);

	// write pixel data
	int num = this->channels * this->width * this->height;

	fwrite(this->pixels, sizeof(unsigned char), num, out);

	fclose(out);
}
