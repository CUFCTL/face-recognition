/**
 * @file ppm.h
 *
 * Interface definitions for a PPM image.
 */
#ifndef PPM_H
#define PPM_H

typedef struct {
	int height;
	int width;
	int max_value;
	unsigned char *pixels;
} ppm_t;

ppm_t * ppm_construct();
void ppm_destruct(ppm_t *image);

void ppm_read(ppm_t *image, const char *path);
void ppm_write(ppm_t *image, const char *path);

#endif
