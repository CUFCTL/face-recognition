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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "grayscale.h"
#include "ppm.h"

#define GREY(v,r,g,b) v = (int)(round(.2989 * r + .5870 * g + .1140 * b))

/*
 * This function takes an image and equalizes the
 * RGB value of each pixel according to the 30% 59%
 * 11% weigh scale.
 */
void grayscale(PPMImage* img)
{
    int intensity, index, i, j;
    for (i = 0; i < img->height; i++) {
        for (j = 0; j < img->width; j++) {
            index = i * img->width + j;
            GREY(intensity,
                    img->pixels[index].r,
                    img->pixels[index].g,
                    img->pixels[index].b);
            memset(&img->pixels[index], intensity, sizeof (Pixel));
        }
    }
    return;
}
//--------------------------------------------------------------------
