/*
 *	pca.cu
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
    eigen_type **eigenfacesT, long int *images, long int *imgsize) {
	
	eigen_type *projectedtrainimages_d, *projectedimages_d, *eigenfacesT_d;
	long int i;

	/* allocate result matrix (no, you HAVE to do this... just try NOT doing it.) */
	(*projectedtrainimages) = (eigen_type *)malloc(sizeof(eigen_type*) * (*images) * (*images-1));
	for(i = 0; i < (*images) * (*images-1); i++) {
		(*projectedtrainimages)[i] = 0;
	}

 	/* Allocate device memory for the matrices */
	cublasAlloc((*images) * (*imgsize), sizeof(eigen_type), (void**)&projectedimages_d);
	cublasAlloc((*images - 1) * (*imgsize), sizeof(eigen_type), (void**)&eigenfacesT_d);
	cublasAlloc((*images) * (*images - 1), sizeof(eigen_type), (void**)&projectedtrainimages_d);
	
	/* Initialize the device matrices with the host matrices */
	cublasSetVector((*images) * (*imgsize), sizeof(eigen_type), (*projectedimages), 1, projectedimages_d, 1);
	cublasSetVector((*images - 1) * (*imgsize), sizeof(eigen_type), (*eigenfacesT), 1, eigenfacesT_d, 1);
	cublasSetVector((*images) * (*images - 1), sizeof(eigen_type), (*projectedtrainimages), 1, projectedtrainimages_d, 1);
	cudaThreadSynchronize();
  	
	/* Performs operation using cublas */
	cublasSgemm('n', 't', (*images), (*images-1), (*imgsize), 1, projectedimages_d, (*images), eigenfacesT_d, (*images-1), 1, projectedtrainimages_d, (*images));
	cudaThreadSynchronize();
	
	if(cublasGetError() != CUBLAS_STATUS_SUCCESS) {
		printf("there was a problem using CUDA/CUBLAS...check your setup!\n");
	}

	/* Read the result back */
	cublasGetVector((*images) * (*images - 1), sizeof(eigen_type), projectedtrainimages_d, 1, (*projectedtrainimages), 1);
	
	cudaThreadSynchronize();
	cublasFree(projectedtrainimages_d);
	cublasFree(eigenfacesT_d);
	cublasFree(projectedimages_d);
}

/*
 *	main
 *
 *	Description: 
 *
 *      THIS FUNCTION CALLS:
 *          LoadTrainingDatabase        (matrix_ops.cu)
 *          calculate_projected_images  (pca.cu)
 *          Recognition                 (pca_host.cu)
 *
 */
int main(int argc, char *argv[]) {

	long int images, imgsize, i;
	int imgloop;                                /* number of images and their size */
	eigen_type  *projectedimages, *eigenfacesT; /* 2D matrices for eigenfaces and projected images read in from MATLAB file */
	eigen_type  *projectedtrainimages;          /* calculated from the read in MATLAB file */
	eigen_type *mean;                           /* 1D matrix of average pixel values from training database also read in from MATLAB file */
	
	int testloop[] = {2,3,19,29,70,745,108,146,182,268}; /* test image numbers present in PCA directory (ppm image files) */
	char inputimage[30];

	LoadTrainingDatabase("eigenfaces_200.txt", &projectedimages, &eigenfacesT, &mean, &images, &imgsize);
	
	/* automatically runs through this many recognitions (speed testing) */
	printf("Enter number of faces to test (1-10): ");
	scanf("%d", &imgloop);
	
	if(imgloop > 10)
	  imgloop = 10;
	else if(imgloop < 1)
	  imgloop = 1;
	
	/* init CUDA library that is primarily used for matrices */
	cublasInit();
	
	calculate_projected_images(&projectedtrainimages, &projectedimages, &eigenfacesT, &images, &imgsize);
 	
	/* compare each test image to the database */
	for(i = 0; i<imgloop; i++) {
	    sprintf(inputimage, "4/test/%d.ppm", testloop[i]);
	    Recognition(inputimage, &mean, &projectedimages, &eigenfacesT, &projectedtrainimages, &images, &imgsize);    
	}

	/* Shutdown */
	cublasShutdown();
	printf("PCA done...\n");
	
	free(projectedimages);
	free(eigenfacesT);
	free(projectedtrainimages);
	
	return(0);
}