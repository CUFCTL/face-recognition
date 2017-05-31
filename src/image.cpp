/**
 * @file image.cpp
 *
 * Implementation of the image type.
 *
 * The following formats are supported:
 * - binary PGM (P5)
 * - binary PPM (P6)
 */
#include <cctype>
#include <fstream>
#include "image.h"
#include "logger.h"

/**
 * Construct an image.
 */
Image::Image()
{
	this->_channels = 0;
	this->_width = 0;
	this->_height = 0;
	this->_max_value = 0;
	this->_pixels = nullptr;
}

/**
 * Destruct an image.
 */
Image::~Image()
{
	delete[] this->_pixels;
}

/**
 * Helper function to skip comments in a PGM/PPM image.
 */
void skip_to_next_value(std::ifstream& file)
{
	char c = file.get();
	while ( c == '#' || isspace(c) ) {
		if ( c == '#' ) {
			while ( c != '\n' ) {
				c = file.get();
			}
		}
		else {
			while ( isspace(c) ) {
				c = file.get();
			}
		}
	}

	file.unget();
}

/**
 * Load an image from a PGM/PPM file.
 *
 * @param path
 */
void Image::load(const std::string& path)
{
	std::ifstream file(path, std::ifstream::in);

	// read image header
	std::string header;
	file >> header;

	// determine image channels
	int channels;

	if ( header == "P5" ) {
		channels = 1;
	}
	else if ( header == "P6" ) {
		channels = 3;
	}
	else {
		log(LL_ERROR, "error: cannot read image \'%s\'\n", path.c_str());
		exit(1);
	}

	skip_to_next_value(file);

	// read image metadata
	int width;
	int height;
	int max_value;

	file >> width;
	skip_to_next_value(file);

	file >> height;
	skip_to_next_value(file);

	file >> max_value;
	file.get();

	// verify that image sizes are equal (if reloading)
	int num1 = channels * width * height;
	int num2 = this->_channels * this->_width * this->_height;

	if ( this->_pixels == nullptr ) {
		this->_pixels = new unsigned char[num1];
	}
	else if ( num1 != num2 ) {
		log(LL_ERROR, "error: unequal sizes on image reload\n");
		exit(1);
	}

	this->_channels = channels;
	this->_width = width;
	this->_height = height;
	this->_max_value = max_value;

	// read pixel data
	file.read(reinterpret_cast<char *>(this->_pixels), num1);

	file.close();
}

/**
 * Save an image to a PGM/PPM file.
 *
 * @param path
 */
void Image::save(const std::string& path)
{
	std::ofstream file(path, std::ofstream::out);

	// determine image header
	std::string header;

	if ( this->_channels == 1 ) {
		header = "P5";
	}
	else if ( this->_channels == 3 ) {
		header = "P6";
	}
	else {
		log(LL_ERROR, "error: cannot write image \'%s\'\n", path.c_str());
		exit(1);
	}

	// write image metadata
	file << header << "\n"
		<< this->_width << " "
		<< this->_height << " "
		<< this->_max_value << "\n";

	// write pixel data
	int num = this->_channels * this->_width * this->_height;

	file.write(reinterpret_cast<char *>(this->_pixels), num);

	file.close();
}
