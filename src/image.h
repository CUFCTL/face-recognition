/**
 * @file image.h
 *
 * Interface definitions for the image type.
 */
#ifndef IMAGE_H
#define IMAGE_H

#include <string>

class Image {
public:
	int channels;
	int width;
	int height;
	int max_value;
	unsigned char *pixels;

	Image();
	~Image();

	void load(const std::string& path);
	void save(const std::string& path);
};

#endif
