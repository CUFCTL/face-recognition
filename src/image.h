/**
 * @file image.h
 *
 * Interface definitions for the image type.
 */
#ifndef IMAGE_H
#define IMAGE_H

#define GREY(p) (0.299 * (p)[0] + 0.587 * (p)[1] + 0.114 * (p)[2])

typedef struct {
	int channels;
	int height;
	int width;
	int max_value;
	unsigned char *pixels;
} image_t;

image_t * image_construct();
void image_destruct(image_t *image);

void image_read(image_t *image, const char *path);
void image_write(image_t *image, const char *path);

#endif
