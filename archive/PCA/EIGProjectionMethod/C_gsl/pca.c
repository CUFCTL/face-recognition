/*==================================================================================================
 *	pca.c
 *
 *  edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *  This file contains
 *      Recognition
 *      main
 *
 *  Description: 
 *
 *  Last edited: Jul. 15, 2013
 *  Edits: 
 *
 */
#include "main.h"

/*==================================================================================================
 *	Recognition
 *
 *	parameters
 *      single pointer, type char       = inputimage
 *      double pointer, type eigen_type = mean
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = projectedtrainimages
 *      single pointer, type int        = images
 *      single pointer, type int        = imgsize
 *      single pointer, type int        = num_faces
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      main    (pca.c)
 *
 */
void Recognition(char *inputimage, eigen_type **mean, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, eigen_type **projectedtrainimages, int *images, int *imgsize, 
    int *num_faces) {
	
	culaStatus status;
	cudaEvent_t start, stop;
	char outputtext[30];
	FILE* record;
	int i, j;
	int min_index;
	float elapsedTime;
	double difference, distance, smallest = 1e20;
	eigen_type 	*projectedtestimage;
	eigen_type	*testimage_normalized;
	PPMImage 	*testimage;
	gsl_vector_view projectedtestimage_v, testimage_normalized_v, mean_v, projectedtrainimages_v;
//  gsl_vector_view eigenfacesT_v, projectedtrainimages_v, mean_v;
	gsl_matrix_view eigenfacesT_v;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	eigenfacesT_v = gsl_matrix_view_array(*eigenfacesT, *imgsize, *num_faces);

	/*  read in test image and convert it to grayscale  */
	testimage = ppm_image_constructor(inputimage);
	grayscale(testimage);
	
	/*  normalize input image by subtracting database mean  */
	testimage_normalized = (eigen_type *)malloc(sizeof(eigen_type) * (*imgsize));
	testimage_normalized_v = gsl_vector_view_array(testimage_normalized, *imgsize);

	/*  gsl_vector_subvector_with_stride only if we rewrote the ppm library to use doubles instead of unsigned chars    */
	for(i = 0; i < *imgsize; i++) {
		/*  the ppm library can obviously be improved, but time is running out for me   */
		testimage_normalized[i] = testimage->pixels[i].r - ((*mean)[i]) +1;
	}
	
	/*  project test image (ProjectedTestImage = eigenfacesT' * NormalizedInputImage)   */
	projectedtestimage = (eigen_type *)malloc(sizeof(eigen_type) * (*images));
	projectedtestimage_v = gsl_vector_view_array(projectedtestimage, (*images));
	
	gsl_blas_dgemv(CblasTrans, 1.0, &eigenfacesT_v.matrix, &testimage_normalized_v.vector, 0.0, &projectedtestimage_v.vector);

	projectedtrainimages_v = gsl_vector_view_array(*projectedtrainimages, *num_faces);
	
	/*  remake the projected train images matrix into a differences matrix  */
	for(i=0;i<*images;i++) {	
		gsl_vector_sub (&projectedtrainimages_v.vector, &projectedtestimage_v.vector);
		gsl_blas_ddot (&projectedtrainimages_v.vector, &projectedtrainimages_v.vector, &distance);
		
		if(distance < smallest) {
			smallest = distance;
			min_index = i;
		}
		
		/*  increment vector along projectedtrainimages matrix  */
		projectedtrainimages_v.vector.data = (*images+projectedtrainimages_v.vector.data);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
//  printf("\nOverall Speed:\t\t\t\t%lf (ms)\n", elapsedTime);
	record = fopen("recognition.txt", "a");
	sprintf(outputtext, "%d, %lf\n", (*images), elapsedTime);
	fprintf(record, outputtext);
	fclose(record);
	
	return;
}


/*==================================================================================================
 *	main
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *      LoadTrainingDatabase    (matrix_ops.c)
 *      Recognition             (pca.c)
 *      
 */
int main(int argc, char *argv[]) {

	int num_faces, images, imgsize, i, imgloop,j;   /*  number of images and their size */
	culaStatus status;
	cudaEvent_t start, stop;
	float elapsedTime;
	FILE* record;
	eigen_type  *projectedimages, *eigenfacesT; /*  2D matrices for eigenfaces and projected images read in from MATLAB file    */
	eigen_type  *projectedtrainimages, *projectedtrainimages2;  /*  calculated from the read in MATLAB file */
	eigen_type *mean;   /*  1D matrix of average pixel values from training database also read in from MATLAB file  */
	
	gsl_matrix_view projectedimages_v, eigenfacesT_v, projectedtrainimages_v, mean_v;
	gsl_matrix_view eigen_vec_view, eigen_vec_view_reduced, final_eigen;
	char inputimage[30];	
	
	if(argc != 2) {
		printf("INVALID... supply a database READ THE SRC!\n");
		return 0;
	}

//  culaInitialize();

	/*  load database file  */
	LoadTrainingDatabase(argv[1], &projectedimages, &eigenfacesT, &mean, &images, &imgsize, &num_faces);

	projectedtrainimages = calloc((images)*(num_faces), sizeof(eigen_type));
	projectedtrainimages2 = calloc((images)*(num_faces),sizeof(eigen_type));    /*  hack!   */
 
	projectedimages_v = gsl_matrix_view_array(projectedimages, imgsize, images);
	eigenfacesT_v = gsl_matrix_view_array(eigenfacesT, imgsize, num_faces);
	projectedtrainimages_v = gsl_matrix_view_array(projectedtrainimages, images, num_faces);
	/*  eigenfaces'*A   */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	/*  technically, this is part of the training algorithm, but it takes HD space, so we time it 
	here and add to the training performance. also, i'm multiplying in opposite order to 
	dramatically improve performance later. final results will be the same  */
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,1.0, &projectedimages_v.matrix, &eigenfacesT_v.matrix, 0.0, &projectedtrainimages_v.matrix);

//  for(i=0;i<10;i++)
//      printf("%le\n", projectedtrainimages[i]);
//  for(i=0;i<10;i++)
//      printf("%le\n", projectedtrainimages[i*images]);

//  return 0;	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
//  printf("\nOverall Speed:\t\t\t\t%lf (ms)\n", elapsedTime);
	record = fopen("training.txt", "a");
	sprintf(inputimage, "%d, %lf\n", (images), elapsedTime);
	fprintf(record, inputimage);
	fclose(record);
	
	/*  compare the test image against each image in database   */
	for(j = 0; j<4;j++) {
	    memcpy(projectedtrainimages2, projectedtrainimages, images*num_faces);
	    sprintf(inputimage, "../Image/Train/ORL_200/%d.ppm\0", 2*(j+1));
	    Recognition(inputimage, &mean, &projectedimages, &eigenfacesT, &projectedtrainimages2, &images, &imgsize, &num_faces);      
	}

	free(projectedimages);
	free(eigenfacesT);
	free(projectedtrainimages);
	
	return(0);
}
