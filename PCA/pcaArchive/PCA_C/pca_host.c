/*
 *	pca_host.c
 *
 *	Description: This file contains functions.
 *      
 *      Recognition
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"
#include "pca.h"

clock_t start, end; // profiling clocks
clock_t startfpga, endfpga;

/*
 *	Recognition
 *
 *	parameters: 
 *      pointer, type char =                inputimage
 *      double pointer, type eigen_type =   mean
 *      double pointer, type eigen_type =   projectedimages
 *      double pointer, type eigen_type =   eigenfacesT
 *      double pointer, type eigen_type =   projectedtrainimages
 *      pointer, type long integer =        images
 *      pointer, type long integer =        imgsize
 *      pointer, type long integer =        num_faces
 *
 *	returns: N/A
 *
 *	Description: This file begins by opening a .ppm image designated by the value in testloop
 *  (in pca.c). It converts the designated image to grayscale. It then "normalizes" the image by
 *  subtracting the database average from the image. It then creates a projected test image by 
 *  multiplying the normalized image to the eigenvectors (same process as for the 
 *  projectedtrainimage in pca.c). Finally, it calculates the "distance" of the projected test 
 *  image with the projected train images and finds the training image with the smallest 
 *  distance from the test image.
 *
 *      THIS FUNCTION CALLS
 *          ppm_image_constructor   (ppm.c)
 *          grayscale               (grayscale.c)
 *
 *      THIS FUNCTION IS CALLED BY
 *          main    (pca.c)
 *
 */
void Recognition(char *inputimage, eigen_type **mean, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, eigen_type **projectedtrainimages, long int *images, 
    long int *imgsize, long int *num_faces) {

	char outputtext[1000];
	long int i, j;
	int min_index;
	eigen_type difference, distance, smallest = 1e20;
	eigen_type 	*projectedtestimage;
	eigen_type	*testimage_normalized;
	PPMImage 	*testimage;
//	struct timeval t0, t1;
//	gettimeofday(&t0, 0);

	/* read in test image and convert it to grayscale */
	testimage = ppm_image_constructor(inputimage);
	grayscale(testimage);
	    
	/* normalize input image by subtracting database mean */
    testimage_normalized = (eigen_type *)malloc(sizeof(eigen_type) * (*imgsize));
	for(i = 0; i < *imgsize; i++) {
		testimage_normalized[i] = testimage->pixels[i].r - ((*mean)[i]) + 1;
	}
	
	/* project test image (ProjectedTestImage = eigenfacesT * NormalizedInputImage) */
	projectedtestimage = (eigen_type *)malloc(sizeof(eigen_type) * (*images));
	for(i = 0; i < *num_faces; i++) {
		projectedtestimage[i] = 0;
		for(j = 0; j < *imgsize; j++) {
			projectedtestimage[i] += (*eigenfacesT)[j*(*num_faces) + i] * testimage_normalized[j];
		}
	}

	for(i = 0; i < *images; i++) {
		distance = 0;
		for(j = 0; j < (*num_faces); j++) {
			difference = (*projectedtrainimages)[j*(*images) + i] - projectedtestimage[j];
			distance += difference*difference;
		}
		
		//if(!PROFILING) printf("distance[%d] = %le\n====\n", i, distance);
		if(distance < smallest) {
			smallest = distance;
			min_index = i;
		}
	}
	
//	gettimeofday(&t1, 0);
//	float elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
//	printf("\nOverall Speed:\t\t\t\t%lf (ms)\n", elapsed/1000.0); //printf("%lf\n", elapsed/1000.0);
	sprintf(outputtext, "\n%s matches image index %d\n\n", inputimage, min_index);
	
	printf("%s", outputtext);
}

