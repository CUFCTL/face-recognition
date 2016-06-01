/*
 *	pca.c
 *
 *	Description: This file contains functions and main for executable "pca."
 *      
 *      calculate_projected_images
 *      main
 *
 */

#ifdef WIN32
#include <windows.h>
#endif
#include "pca.h"

/*
 *	calculate_projected_images
 *
 *	parameters:
 *      double pointer, type eigen_type =   projectedtrainimages
 *      double pointer, type eigen_type =   projectedimages
 *      double pointer, type eigen_type =   eigenfacesT
 *      pointer, type long integer =        images
 *      pointer, type long integer =        imgsize
 *      pointer, type long integer =        num_faces
 *
 *	returns: N/A
 *      implicitly returns values for
 *          projectedtrainimages
 *
 *	Description: This function initializes the variable "projectedtrainingimages" using two arrays
 *  and three variables (initialized in matrix_ops.c by function LoadTrainingDatabase). This 
 *  function multiplies each eigenvector in the covariance matrix by every image in the database
 *  and stores the result in "projectedtrainingimages."
 *      projectedtrainimages - an array of size num_faces X images
 *          This array is the product eigenvector covariance matrix and the matrix of centered
 *          images.
 *
 *      THIS FUNCTION CALLS
 *
 *      THIS FUNCTION IS CALLED BY
 *          main    (pca.c)
 *
 */
void calculate_projected_images(eigen_type **projectedtrainimages, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, long int *images, long int *imgsize, long int *num_faces) {

	long int i, j, k;
	
	/* projected images from training database (projectedtrainimages = eigenfacesT * projectedimages) */
	(*projectedtrainimages) = (eigen_type *)malloc(sizeof(eigen_type*) * (*images) * (*num_faces));

	for(i = 0; i < *num_faces; i++) {
		for(k = 0; k < *images; k++) {
			(*projectedtrainimages)[i*(*images) + k] = 0;
			
			for(j = 0; j < *imgsize; j++) {
				(*projectedtrainimages)[i*(*images) + k] += (*eigenfacesT)[j*(*num_faces) + i] * (*projectedimages)[j*(*images) + k];
			}
 	    }
	}
}

/*
 *	main
 *
 *	Description: This main function calls several other functions in order to implement 
 *  the PCA algorithm.
 * 
 *      THIS FUNCTION CALLS
 *          LoadTrainingDatabase        (matrix_ops.c)
 *          calculate_projected_images  (pca.c)
 *          Recognition                 (pca_host.c)
 *
 */
int main(int argc, char *argv[]) {

	long int num_faces, images, imgsize, i, imgloop;    /* number of images and their size */
	eigen_type  *projectedimages, *eigenfacesT;         /* 2D matrices for eigenfaces and projected images read in from MATLAB file */
	eigen_type  *projectedtrainimages;                  /* calculated from the read in MATLAB file */
	eigen_type *mean;                                   /* 1D matrix of average pixel values from training database also read in from MATLAB file */
	
	/* test image numbers present in PCA directory (ppm image files) */
	int testloop[] = {2,3,19,29,70,745,108,146,182,268}; 
	char inputimage[30];

    /* Calling LoadTrainingDatabase will return values for projectedimages, eigenfacesT, mean, images, imgsize, and num_faces  */
	LoadTrainingDatabase("eigenfaces_200.txt", &projectedimages, &eigenfacesT, &mean, &images, &imgsize, &num_faces);
        
	/* automatically runs through this many recognitions (speed testing) */
	printf("Enter number of faces to test (1-10): ");
	scanf("%ld", &imgloop);
	
	if(imgloop > 10)
	    imgloop = 10;
	else if(imgloop < 1)
	    imgloop = 1;
	
	/* Calling calculate_projected_images will return values for projectedtrainimages */
	/* do this once (most time consuming) */
	calculate_projected_images(&projectedtrainimages, &projectedimages, &eigenfacesT, &images, &imgsize, &num_faces);

	/* compare the test image against each image in database */
	for(i = 0; i<imgloop; i++) {
	    sprintf(inputimage, "4/test/%d.ppm", testloop[i]);
	    Recognition(inputimage, &mean, &projectedimages, &eigenfacesT, &projectedtrainimages, &images, &imgsize, &num_faces);
	        printf("recognition iteration %ld finished...\n", i+1);
	}

	printf("PCA done...\n");

	free(projectedimages);
	free(eigenfacesT);
	free(projectedtrainimages);

	return(0);
}
