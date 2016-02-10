/*
 *	matrix_ops.cu
 *
 *	Description: This file contains functions used to implement PCA with CUDA
 *
 *      LoadTrainingDatabase
 *
 */
 
#include "pca.h"

/* fill the three matrices above with data from the MATLAB file */

/*
 *	LoadTrainingDatabase
 *
 *	parameters: 
 *      pointer, type character         = filename
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = mean
 *      pointer, type long integer      = images
 *      pointer, type long integer      = imgsize
 *
 *	returns: N/A
 *      Implicitly returns values for
 *          projectedimages
 *          eigenfacesT
 *          mean
 *          images
 *          imgsize
 *
 *	Description: This function initializes key variables that will be used throughout the 
 *  implementation of the CUDA code. These variables are read in from a file that is created by a 
 *  Matlab script and are the following
 *      projectedimages - a matrix of the aligned images
 *      eigenfacesT - a matrix of eigenvectors of the covariance matrix
 *      mean - the "average" of all the training images
 *      images - the number of images contained in the file
 *      imgsize - the size of all the images in pixels
 *
 *      THIS FUNCTION CALLS
 *
 *      THIS FUNCTION IS CALLED BY
 *          main    (pca.cu)
 *
 */
void LoadTrainingDatabase(char *filename, eigen_type **projectedimages, eigen_type **eigenfacesT, 
    eigen_type **mean, long int *images, long int *imgsize) {
	
	long int i, j;
	double temp;
	long int sizeW, sizeH;

	FILE *f = fopen(filename, "r");
   
	if (f == NULL) { printf("Failed to open eigenfaces file: %s\n", filename); return; }
   
	printf("opening %s...\n", filename);
	fflush(stdout);
   
	/* first lines of file contains the number of images and their size in pixels */
	/* read in the number of images */
	fscanf(f, "%le", &temp);
	*images = (int)temp;
	
	/* added to read an extra arbitrary value from the Matlab generated file */
	fscanf(f, "%le", &temp);
	
	/* read in the size of the images in pixels */
	fscanf(f, "%le", &temp);
	*imgsize = (int)temp;
   
	printf("Database contains %ld images...\n", *images);
	fflush(stdout);
   
	/* read eigenfaces */
	sizeW = *images;
	sizeH = *imgsize;
	
	(*eigenfacesT) = (eigen_type *)malloc(sizeH*(sizeW-1)*sizeof(eigen_type));
	j = 0;
	i = 0;

	while(!feof(f)) {
		if(i >= sizeW-1) {
			if(++j == sizeH) break;
         
			i = 0;
		}

		fscanf(f, "%le", &temp);
		(*eigenfacesT)[j*(sizeW-1) + i] = (eigen_type)temp;
		i++;
	}
   
	/* read projected images */
	sizeW = *images;

	(*projectedimages) = (eigen_type *)malloc(sizeH*sizeW*sizeof(eigen_type));
	j = 0; 
	i = 0;

	while(!feof(f)) {
		if(i >= sizeW) {
			if(++j == sizeH) break;
			i = 0;
		}

		fscanf(f, "%le", &temp);
		(*projectedimages)[j*sizeW+i] = (eigen_type)temp;
		i++;
	}
  
	/* read mean */
	sizeW = *imgsize;
	sizeH = 1;
   
	(*mean) = (eigen_type *)malloc(sizeW*sizeof(eigen_type));

	j = 0; 
	i = 0; 

	while(!feof(f)) {
		if(i >= sizeW) break;

		fscanf(f, "%le", &temp);
		(*mean)[i] = (eigen_type)temp;
		i++;
	}
   
	fclose(f);  
}