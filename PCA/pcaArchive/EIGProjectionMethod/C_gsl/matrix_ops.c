/*==================================================================================================
 *	matrix_ops.c
 *
 *  Edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *  THIS FILE CONTAINS
 *      LoadTrainingDatabase
 *
 *	Description: 
 *
 *  Last edited: Jul. 25, 2013
 *  Edits:
 *
 */
#include "main.h"

/*  fill the three matrices above with data from the MATLAB file    */
/*==================================================================================================
 *  LoadTrainingDatabase
 *
 *  parameters
 *      single pointer, type char       = filename
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = mean
 *      single pointer, type int        = images
 *      single pointer, type int        = imgsize
 *      single pointer, type int        = facessize
 *
 *  returns
 *      N/A
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      main    (pca.c)
 */
void LoadTrainingDatabase(char *filename, eigen_type **projectedimages, eigen_type **eigenfacesT, 
    eigen_type **mean, int *images, int *imgsize, int *facessize) {
	
	int i, j;
	int tempi;
	double tempf;
	int sizeW, sizeH;

	FILE *f = fopen(filename, "rb");
   
	if (f == NULL) { printf("Failed to open eigenfaces file: %s\n", filename); return; }
   
	printf("opening %s...\n", filename);
	fflush(stdout);
   
	/*  first line of file contains the number of images and their size in pixels   */
	fread(&tempi, sizeof(int), 1, f);
	*images = tempi;
	
	fread(&tempi, sizeof(int), 1, f);
	*facessize = tempi; 
	
	fread(&tempi, sizeof(int), 1, f);
	*imgsize = tempi;
   
	/*  read eigenfaces */
	sizeW = *facessize;
	sizeH = *imgsize;

	printf("Database contains %d images...\n", *images);
	fflush(stdout);
   
	(*eigenfacesT) = (eigen_type *)malloc(sizeH*(sizeW)*sizeof(eigen_type));
	j = 0;  /*  row */
	i = 0;  /*  column  */

	while(!feof(f)) {
		if(i >= sizeW) {
			if(++j == sizeH) break;
			
			i = 0;
		}

//      temp = rand();
		fread(&tempf, sizeof(double), 1, f);
		(*eigenfacesT)[j*(sizeW) + i] = tempf;
		i++;
	}
   
	/*  read projected images   */
	sizeW = *images;
	sizeH = *imgsize;

	(*projectedimages) = (eigen_type *)malloc(sizeH*sizeW*sizeof(eigen_type));
	j = 0; 
	i = 0;

	while(!feof(f)) {
		if(i >= sizeW) {
			if(++j == sizeH) break;
			
			i = 0;
		}

//      temp = rand();
		fread(&tempf, sizeof(double), 1, f);
		(*projectedimages)[j*sizeW+i] = tempf;
		i++;
	}
  
	/*  read mean   */
	sizeW = *imgsize;
	sizeH = 1;
   
	(*mean) = (eigen_type *)malloc(sizeW*sizeof(eigen_type));

	j = 0; 
	i = 0; 

	while(!feof(f)) {
		if(i >= sizeW) break;
      
//      temp = rand();
		fread(&tempf, sizeof(double), 1, f);
		(*mean)[i] = tempf;
		i++;
	}
   
	fclose(f);
	
	return;	
}
