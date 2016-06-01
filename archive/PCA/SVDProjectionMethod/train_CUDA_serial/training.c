/*==================================================================================================
 *  training.c
 *
 *  Edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *  THIS FILE CONTAINS
 *      main
 *
 *  Description: 
 *
 *  Last edited: Jul. 22, 2013
 *  Edits: 
 *
 */
#include "main.h"
#define DEBUG
#define precision float

/*==================================================================================================
 *  main
 *
 *  Description: 
 *
 *  THIS FUNCTION CALLS
 *
 */
int main() {
	culaStatus status;
	cudaEvent_t start, stop;
	float elapsedTime;
	int i, j, k;
	char file_path[] = "../../Image/Train/ORL_200/";
	char full_path[100];
	int width, height;
	FILE *input_matrix;
	FILE *eigen_file;
	PPMImage 	*testimage;
	precision *a, *singular, *singular_m, *a_copy, *b, *eigenvector, *c;	
	precision *mean, *eigenvalue;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	culaInitialize();
	
	//get size of image
	sprintf(full_path, "%s%d.ppm", file_path, 1);
	testimage = ppm_image_constructor(full_path);
	width = testimage->width;
	height = testimage->height;
	ppm_image_destructor(testimage, 1);

	a = calloc(width*height*number_copies*existing_images, sizeof(precision));
	singular = calloc(number_copies*existing_images, sizeof(precision));
	singular_m = calloc(number_copies*existing_images*number_copies*existing_images, sizeof(precision));
	a_copy = calloc(width*height*number_copies*existing_images, sizeof(precision));
	b = calloc(number_copies*existing_images*number_copies*existing_images, sizeof(precision));
	c = calloc(width*height*number_copies*existing_images, sizeof(precision));	

	eigenvector = calloc(number_copies*existing_images*number_copies*existing_images, sizeof(precision));
	eigenvalue = calloc(number_copies*existing_images, sizeof(precision));
	mean = calloc(width*height, sizeof(precision));

	input_matrix = fopen("T.txt", "r");
	double temp;
	for(k = 0; k < width*height; k++)
	for(i = 0; i < number_copies; i++)
	{
		for(j = 0; j < existing_images; j++)
		{

			fscanf(input_matrix, "%lf", &temp);
			a[k*number_copies*existing_images+(i+1)*j] = temp;
			mean[k] += temp;
/*
			sprintf(full_path, "%s%d.ppm", file_path, j+1);
			testimage = ppm_image_constructor(full_path);
			grayscale(testimage);
			//copy image into the first matrix
			for(k = 0; k < width*height; k++)
			{
				a[number_copies*existing_images * k + (i+1)*j] = testimage->pixels[k].r;
				mean[k] += testimage->pixels[k].r;
			}
			ppm_image_destructor(testimage, 1);	*/
		}
	}
	fclose(input_matrix);
	
	printf("Images loaded...\n");
	fflush(stdout);
	for(i = 0; i < width*height; i++)
	{
		mean[i] /= existing_images*number_copies;
	}
	//subtract the mean from the elements of matrix a
	for(i = 0; i < number_copies*existing_images; i++)
	{
		for(j = 0; j < width*height; j++)
		{
			//gsl_matrix_set (a, j, i, gsl_matrix_get(a, j,i) - mean->data[j]);//testimage->pixels[k].r);
			a[number_copies*existing_images*j + i] -= mean[j];
			a_copy[number_copies*existing_images*j + i] = a[number_copies*existing_images*j + i];
		}
	}
	printf("Mean subtracted, images database normalized...\n");
	fflush(stdout);
	//find covariance matrix "b"

//[U,S,V] = svd(A, 0);
//%produces the "economy size"
//%decomposition. If X is m-by-n with m > n, then only the
//%first n columns of U are computed and S is n-by-n.
//%For m <= n, SVD(X,0) is equivalent to SVD(X).

//Covariance_matrix =V*(S^2)*V';

//Eigenfaces = A * Covariance_matrix; % A: centered image vectors
	status = culaSgesdd('S', width*height, number_copies*existing_images, a, width*height, singular, a_copy, width*height, b, number_copies*existing_images); 
//	status = culaSgesvd('N', 'S', width*height, number_copies*existing_images, a, width*height, singular, a_copy, width*height, b, number_copies*existing_images); 
	#ifdef DEBUG
	culaGetErrorInfoString(status, culaGetErrorInfo(), full_path, 100);
	printf("1st step: %s\n", full_path);
	#endif
	for(i =0; i<200; i++)
		printf("%le\n", singular[i]);	
	//put the values into the diagonal of the properly sized matrix
	for(i=0;i<number_copies*existing_images;i++)
	{
		singular_m[i*number_copies*existing_images +i] = singular[i];
	}

	//square the single matrix
	status = culaSgemm('N', 'N',number_copies*existing_images,number_copies*existing_images, number_copies*existing_images, 1.0, singular_m,number_copies*existing_images, singular_m, number_copies*existing_images, 0.0, singular_m, number_copies*existing_images);		
	//status = culaSgemm('',number_copies*existing_images,number_copies*existing_images, number_copies*existing_images, 1.0, singular_m,number_copies*existing_images, singular_m, number_copies*existing_images, 0.0, singular_m, number_copies*existing_images);		

	#ifdef DEBUG
	culaGetErrorInfoString(status, culaGetErrorInfo(), full_path, 100);
	printf("2nd step: %s\n", full_path);	
	#endif

	//multiply by Vector
	status = culaSgemm('T', 'N',number_copies*existing_images, number_copies*existing_images, number_copies*existing_images, 1.0, b,number_copies*existing_images, singular_m, number_copies*existing_images, 0.0, singular_m, number_copies*existing_images);		
	#ifdef DEBUG
	culaGetErrorInfoString(status, culaGetErrorInfo(), full_path, 100);
	printf("3rd step: %s\n", full_path);
	#endif

	//multiply by Vector transpose
	status = culaSgemm('N', 'N',number_copies*existing_images, number_copies*existing_images, number_copies*existing_images, 1.0, singular_m,number_copies*existing_images, b, number_copies*existing_images, 0.0, singular_m, number_copies*existing_images);		
	#ifdef DEBUG
	culaGetErrorInfoString(status, culaGetErrorInfo(), full_path, 100);
	printf("4th step: %s\n", full_path);
	#endif
	for(i = 0; i<10;i++)
		printf("%le\n", singular_m[i]);
	eigen_file = fopen("eigen.txt", "wb");
	i = number_copies*existing_images;
	fwrite(&i, sizeof(int), 1, eigen_file);
	fwrite(&i, sizeof(int), 1, eigen_file);
	i = width*height;
	fwrite(&i, sizeof(int), 1, eigen_file);
	fwrite(c, sizeof(precision), width*height*number_copies*existing_images, eigen_file);
	fwrite(a, sizeof(precision), width*height*number_copies*existing_images, eigen_file);
	fwrite(mean, sizeof(precision), width*height, eigen_file);
	fclose(eigen_file);

	culaShutdown();

	free(a);
	free(a_copy);
	free(b);
	free(c);
	free(eigenvector);
	free(eigenvalue);
	free(mean);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\nOverall Speed:\t\t\t\t%lf (ms)\n", elapsedTime);

	return 0;
}
