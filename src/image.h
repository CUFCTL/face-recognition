/**
 * @file image.h
 *
 * Interface definitions for the image type.
 */
#ifndef IMAGE_H
#define IMAGE_H

#include <string>

class Image {
private:
	int _channels;
	int _width;
	int _height;
	int _max_value;
	unsigned char *_pixels;

public:
	Image();
	~Image();

	inline int channels() const { return this->_channels; }
	inline int width() const { return this->_width; }
	inline int height() const { return this->_height; }
	inline int max_value() const { return this->_max_value; }
	inline unsigned char& elem(int i) const { return this->_pixels[i]; }

	void load(const std::string& path);
	void save(const std::string& path);
};

#endif
