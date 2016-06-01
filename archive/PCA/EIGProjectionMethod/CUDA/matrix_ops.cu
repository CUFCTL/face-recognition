/*==================================================================================================
 *	matrix_ops.cu
 *
 *  Edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *  THIS FILE CONTAINS
 *      LoadTrainingDatabase
 *      
 *	Description: 
 *
 *  Last edited: Jul. 18, 2013
 *  Edits: 
 *
 */
#include "pca.h"

/*  fill the three matrices above with data from the MATLAB file    */
/*==================================================================================================
 *	LoadTrainingDatabase
 *
 *	parameters
 *      single pointer, type char       = filename
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = mean
 *      single pointer, type long int   = images
 *      single pointer, type long int   = imgsize
 *      single pointer, type long int   = facessize
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      main    (pca.cu)
 *
 */
void LoadTrainingDatabase(char *filename, eigen_type **projectedimages, eigen_type **eigenfacesT, 
    eigen_type **mean, long int *images, long int *imgsize, long int *facessize) {
	
	long int i, j;
	double temp;
	long int sizeW, sizeH;

	FILE *f = fopen(filename, "r");
   
	if (f == NULL) { printf("Failed to open eigenfaces file: %s\n", filename); return; }
	printf("opening %s...\n", filename);
	fflush(stdout);
   
	/*  first line of file contains the number of images and their size in pixels   */
	fscanf(f, "%le", &temp);
	*images = (int)temp;
   
	fscanf(f, "%le", &temp);
	*facessize = (int)temp;
   
	fscanf(f, "%le", &temp);
	*imgsize = (int)temp;
   
	/*  read eigenfaces */
	sizeW = *facessize;
	sizeH = *imgsize;

	printf("Database contains %ld images...\n", *images);
	fflush(stdout);
   
	(*eigenfacesT) = (eigen_type *)malloc(sizeH*(sizeW)*sizeof(eigen_type));
	j = 0;  /*  row */
	i = 0;  /*  column  */

	while(!feof(f)) {
		if(i >= sizeW) {
			if(++j == sizeH)    break;
		
			i = 0;
		}

//      temp = rand();
		fscanf(f, "%le", &temp);
		(*eigenfacesT)[j*(sizeW) + i] = (eigen_type)temp;
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
			if(++j == sizeH)    break;
         
			i = 0;
		}

//      temp = rand();
		fscanf(f, "%le", &temp);
		(*projectedimages)[j*sizeW+i] = (eigen_type)temp;
		i++;
	}
  
	/*  read mean   */
	sizeW = *imgsize;
	sizeH = 1;
   
	(*mean) = (eigen_type *)malloc(sizeW*sizeof(eigen_type));

	j = 0; 
	i = 0; 

	while(!feof(f)) {
		if(i >= sizeW)  break;
      
//      temp = rand();
		fscanf(f, "%le", &temp);
		(*mean)[i] = (eigen_type)temp;
		i++;
	}
	fclose(f);
	
	return;
}
