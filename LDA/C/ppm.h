/*
   PPMShop (C) 2008 Josiah s. Yeagley aka meanmon13

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; see the file COPYING; if not, write to the
   Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
   MA 02110-1301 USA
 */

#include <stdio.h>

#ifndef __PPM_H__
#define __PPM_H__

/*
 * Uses the standard RGB format for representing
 * the color of each pixel in the .ppm file format
 */
typedef struct {
    union { // When converted to grayscale the red channel & intensity are one
            // and the same
        unsigned char r; // red value
        unsigned char intensity; // 8-bit value: 0x00 = white, 0xFF = black
    };
    unsigned char g; // green value
    unsigned char b; // blue value
} Pixel;

/*
 * Contains details about the image as well as a pointer to it.
 */
typedef struct {
    unsigned char p;        // ppm identifier(P3 or P6)
    unsigned int width;     // width of the image in pixels
    unsigned int height;    // height of the image in pixels
    unsigned int maxValue;  // maximum pixel value
    Pixel* pixels;          // starting address for the image buffer
    int size;               // Size of image on disk in bytes
    char filename[100];     // filename of the image
} PPMImage;

// Member functions
PPMImage* ppm_image_constructor(const char* filename);
void ppm_image_destructor(PPMImage*, char delete_flag);
PPMImage* load_ppm_image(PPMImage* img, const char* filename);
void read_ppm_header(PPMImage*, FILE* in);
PPMImage* read_P3_to_P6(PPMImage*, FILE* in);
void skip_to_next_value(FILE*);

// grayscale.c
void grayscale(PPMImage*);

#endif
