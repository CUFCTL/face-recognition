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

/*
   CHANGE LOG
   2013.10.13 - Jesse Tetreault
   - Added fclose(in) to load_ppm_image
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//#include <malloc.h>
#include <errno.h>

#include "ppm.h"

#define ERROR_OUT stderr


/*
 * This is the constructor for the struct PPMImage. this_prt is passed
 * to the constructer as NULL when a new instance of IMMImage needs to
 * be declared on the heap. this_ptr contains a valid pointer if the
 * memory for the object has already been allocated. May be called
 * with only this_ptr or can be called using a filename as the second
 * parameter in which case the imge will be loaded.
*/
PPMImage* ppm_image_constructor(const char * filename)
{
   //Allocate memory for a new object on the heap
   PPMImage* this_ptr = (PPMImage *) malloc(sizeof(PPMImage));
   if(!this_ptr)
   {
		fprintf(ERROR_OUT, "ERROR %d: Unable to allocate memory of PPMImage Object\nPress Enter To Exit...", errno);
		getc(stdin);
        exit(10);
   }

   //If memory has been allocated initialize the member variables
   if(this_ptr)
   {
      this_ptr->p = '\0';
      this_ptr->width = 0;
      this_ptr->height = 0;
      this_ptr->maxValue = 0;
      this_ptr->size = 0;
      this_ptr->filename[0] = '\0';
   }

   //If file name was specified then load the image
   if(filename)
   {
      this_ptr = load_ppm_image(this_ptr, filename);
   }


return this_ptr;
}
//--------------------------------------------------------------------

/*
 * This function acts as a destructor for the PPMImage struct.
 * It is here for memory management purposes. A pointer to the struct
 * and a flag are passed so that if the flag is passed as true
 * the object is also deleted from the heap.
*/
void ppm_image_destructor(PPMImage* this_ptr, char delete_flag)
{
  //Remove the image from the heap
  if(this_ptr->pixels)
  {
	   free(this_ptr->pixels);
  }

   //if the object was declared on the heap free it.
   if(delete_flag)
   {
      free(this_ptr);
   }

return;
}
//--------------------------------------------------------------------

/*
 * This fuction loads the given ppm file
 * and returns a pointer to it
*/
PPMImage* load_ppm_image(PPMImage* img, const char* filename)
{

  //printf("Loading file: %s\n",filename);

  if(strlen(filename) <= 3)
  {
    fprintf(ERROR_OUT, "ERROR: Invalid filename. %s \nPress Enter To Exit...",filename);
	getc(stdin);
    exit(20);
  }

  FILE* in;

  in = fopen(filename,"rb");
  if(in == NULL)
  {
     fprintf(ERROR_OUT, "ERROR %d: Unable to read file.\nPress Enter To Exit...", errno);
	 getc(stdin);
     exit(30);
  }

  strcpy(img->filename, filename);

  read_ppm_header(img, in);

  //Allocate memory on the heap for the image array.
  img->size = img->width * img->height;
  img->pixels = (Pixel*) malloc(img->size * sizeof(Pixel));

  if(img->p == '6')
  {
     fread(img->pixels, sizeof(Pixel), img->size, in);
  }
  else if(img->p == '3')
  {
     read_P3_to_P6(img, in);
  }
  
  //ADDED LINE INTO ppm.c
  fclose(in);

return img;
}
//--------------------------------------------------------------------

/*
 * Reads the header of the given ppm image file
*/
void read_ppm_header(PPMImage* img, FILE *in)
{
   char ch = fgetc(in); //skip over 'P'
   img->p = fgetc(in);

   if(ch != 'P')
   {
      fprintf(ERROR_OUT, "ERROR: Invalid PPM Image.\nPress Enter To Exit...");
	  getc(stdin);
      exit(40);
   }
   else if(img->p != '3' && img->p != '6')
   {
      fprintf(ERROR_OUT, "ERROR: Invalid PPM Identifier %c.\nPress Enter To Exit...",img->p);
	  getc(stdin);
      exit(40 + img->p);
   }

   skip_to_next_value(in);
   fscanf(in, "%d", &img->width);

   skip_to_next_value(in);
   fscanf(in, "%d", &img->height);

   skip_to_next_value(in);
   fscanf(in, "%d", &img->maxValue);

   skip_to_next_value(in);

   //printf("width: %d, height: %d\n", img->width, img->height);

return;
}
//--------------------------------------------------------------------


/*
 * This function reads PPM images of type P3
 * while converting it to a type P6
*/
PPMImage* read_P3_to_P6(PPMImage* img, FILE* in)
{
   printf("Your image is type P3. Convertering it to type P6.\n");
   printf("This process may take longer then normal");

   img->p = '6';
   int i,j;
   for(i = 0; i < img->height; ++i)
   {
      for(j = 0; j < img->width; ++j)
      {
         int index = i * img->width + j;
         fscanf(in, "%u", (unsigned int *) &(*((*img).pixels+index)).r);
         fscanf(in, "%u", (unsigned int *) &(*((*img).pixels+index)).g);
         fscanf(in, "%u", (unsigned int *) &(*((*img).pixels+index)).b);
      }

      //Progress indicator
      if(i % 100 == 0)
      {
         printf(".");
      }
   }
   printf("\n");

return img;
}
//--------------------------------------------------------------------

/*
 * Skips over irrelevant data; namely, comments & spaces
*/
void skip_to_next_value(FILE* in)
{
   char ch = fgetc(in);
   while(ch == '#' || isspace(ch))
   {
       if(ch == '#')
       {
          while(ch != '\n')
          {
             ch = fgetc(in);
          }
       }
       else
       {
          while(isspace(ch))
          {
             ch = fgetc(in);
          }
       }
   }

   ungetc(ch,in); //return last read value

return;
}
//--------------------------------------------------------------------

