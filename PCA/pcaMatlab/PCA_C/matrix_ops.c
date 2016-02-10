/*
 *	matrix_ops.c
 *
 *	Description: This file contains functions.
 *      
 *      LoadTrainingDatabase
 *
 */


#include "pca.h"


// fill the three matrices above with data from the MATLAB file

/*
 *	LoadTrainingDatabase
 *
 *	parameters: 
 *      pointer, type char =                filename
 *      double pointer, type eigen_type =   projectedimages
 *      double pointer, type eigen_type =   eigenfacesT
 *      double pointer, type eigen_type =   mean
 *      pointer, type long integer =        images
 *      pointer, type long integer =        imgsize
 *      pointer, type long integer =        facessize
 *
 *	returns: N/A
 *      implicitly returns values for
 *          projectedimages
 *          eigenfacesT
 *          mean
 *          images
 *          facessize
 *          imgsize
 *
 *	Description: This function initializes several variables that are used to implement the 
 *  executable pca. All of the data needed to fill the variables comes from a Matlab file whose name
 *  is contained in the variable "filename."
 *      projectedimages - an array of size imgsize X images
 *          Matrix of centered image vectors
 *      eigenfacesT - an array of size imgsize X facessize
 *          Eigen vectors of the covariance matrix of the training database
 *      mean - an array of size 1 X imgsize
 *          Mean of the training database
 *      images - number of images held in file "eigenfaces_100.txt"
 *      imgsize - size, in pixels, of an image in "eigenfaces_100.txt"
 *      facessize - number of different faces in file "eigenfaces_100.txt"
 *
 *      THIS FUNCTION CALLS
 *
 *      THIS FUNCTION IS CALLED BY
 *          main    (pca.c)
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
   
	/* first lines of file contains the number of images and their size in pixels */
	/* read in the number of images */
	fscanf(f, "%le", &temp);
	*images = (int)temp;
	
	/* read in the number of different faces */
	fscanf(f, "%le", &temp);
	*facessize = (int)temp;
	
	/* read in the size of the images in pixels */
	fscanf(f, "%le", &temp);
	*imgsize = (int)temp;
   
    printf("Database contains %ld images...\n", *images);
	fflush(stdout);
	


	/* read in eigenvectors of covariance matrix */
	sizeW = *facessize;
	sizeH = *imgsize;

	(*eigenfacesT) = (eigen_type *)malloc(sizeH*(sizeW)*sizeof(eigen_type));
	j = 0;
	i = 0;

	while(!feof(f)) {
		if(i >= sizeW) {
			if(++j == sizeH) break;
		i = 0;
		}

		fscanf(f, "%le", &temp);
		(*eigenfacesT)[j*(sizeW) + i] = (eigen_type)temp;
		
		i++;
	}
   
	/* read matrix of centered images */
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
