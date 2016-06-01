/*==================================================================================================
 *	pca.c
 *
 *  Edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *	Description: 
 *
 *  Last edited: Jul. 18, 2013
 *  
 *  THIS FILE CONTAINS
 *      calculate_projected_images
 *      main
 *
 */
#ifdef WIN32
#include <windows.h>
#endif
#include "pca.h"

/*==================================================================================================
 *	calculate_projected_images
 *
 *	parameters: 
 *	returns: 
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      main
 */
void calculate_projected_images(eigen_type **projectedtrainimages, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, long int *images, long int *imgsize, long int *num_faces) {
	
	long int i, j, k;
	
	/*  projected images from training database (ProjectedImages = eigenfacesT * projectedimages)   */
	(*projectedtrainimages) = (eigen_type *)malloc(sizeof(eigen_type*) * (*images) * (*num_faces));

	for(i = 0; i < *num_faces; i++) {
		for(k = 0; k < *images; k++) {
			(*projectedtrainimages)[i*(*images) + k] = 0;
			
			for(j = 0; j < *imgsize; j++)   (*projectedtrainimages)[i*(*images) + k] += (*eigenfacesT)[j*(*num_faces) + i] * (*projectedimages)[j*(*images) + k];
		}
	}
}

/*==================================================================================================
 *	main
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *      LoadTrainingDatabase
 *      calculate_projected_images
 *      Recognition
 */
int main(int argc, char *argv[]) {

	long int num_faces, images, imgsize, i, imgloop;    /*  number of images and their size */
	eigen_type  *projectedimages, *eigenfacesT; /*  2D matrices for eigenfaces and projected images read in from MATLAB file    */
	eigen_type  *projectedtrainimages;  /*  calculated from the read in MATLAB file */
	eigen_type *mean;   /*  1D matrix of average pixel values from training database also read in from MATLAB file  */
	
	int testloop[] = {2,3,19,29,70,745,108,146,182,268};    /*  test image numbers present in PCA directory (ppm image files)   */
	char inputimage[30];

	LoadTrainingDatabase("eigenfaces.txt", &projectedimages, &eigenfacesT, &mean, &images, &imgsize, &num_faces);

	/*  automatically runs through this many recognitions (speed testing)   */
	printf("Enter number of faces to test (1-10): ");
	scanf("%d", &imgloop);
	
	if(imgloop > 10)    imgloop = 10;
	else if(imgloop < 1)    imgloop = 1;
	
	/*  do this once (most time consuming)  */
	calculate_projected_images(&projectedtrainimages, &projectedimages, &eigenfacesT, &images, &imgsize, &num_faces);

	/*  compare the test image against each image in database   */
	for(i = 0; i<imgloop; i++) {
	    sprintf(inputimage, "4/test/%d.ppm", testloop[i]);
	    Recognition(inputimage, &mean, &projectedimages, &eigenfacesT, &projectedtrainimages, &images, &imgsize, &num_faces);    
	}

	printf("PCA done...\n");

	free(projectedimages);
	free(eigenfacesT);
	free(projectedtrainimages);
	return(0);
}
