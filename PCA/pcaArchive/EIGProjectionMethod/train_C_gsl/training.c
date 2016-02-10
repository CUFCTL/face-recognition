#include "main.h"
//#include "cblas.h"

#define precision double

int main(int argn, const char* argv[]) {
	culaStatus status;
	cudaEvent_t start, stop;
	int i, j, k, l;
	float elapsedTime;
	char file_path[] = "../Image/Train/ORL_200/";
	char full_path[100];
	int width, height;
	FILE *input_matrix;
	FILE *eigen_file;
	PPMImage 	*testimage;
	precision *a, *b, *eigenvector, *c, *eigenvector_thr,  *a_copy;	
	precision *mean, *eigenvalue, *eigenvalue_thr, *image_assignment;
	gsl_eigen_symmv_workspace *w;
	gsl_matrix_view A, B, eigen_vec_view, eigen_vec_view_reduced, final_eigen;// = gsl_matrix_view_array(a, 2, 3);
	gsl_vector_view  eigen_val_view;
	
	if (argn < 2)
		return -1;
	
	int number_copies = atoi(argv[1]);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	culaInitialize();

    /*  get size of image   */
	sprintf(full_path, "%s%d.ppm", file_path, 1);
	testimage = ppm_image_constructor(full_path);
	width = testimage->width;
	height = testimage->height;
	ppm_image_destructor(testimage, 1);

	a = calloc(number_copies*existing_images*width*height, sizeof(precision));
	b = calloc(number_copies*existing_images*number_copies*existing_images, sizeof(precision));
	c = calloc (width*height*number_copies*existing_images, sizeof(precision));
	
	A = gsl_matrix_view_array_with_tda(a, width*height, number_copies*existing_images,number_copies*existing_images);
	B = gsl_matrix_view_array(b, number_copies*existing_images, number_copies*existing_images);

	eigenvector = calloc(number_copies*existing_images*number_copies*existing_images, sizeof(precision));
	eigenvalue = calloc(number_copies*existing_images, sizeof(precision));

	eigen_vec_view = gsl_matrix_view_array_with_tda(eigenvector, number_copies*existing_images, number_copies*existing_images, number_copies*existing_images);
	eigen_val_view = gsl_vector_view_array(eigenvalue, number_copies*existing_images);
	final_eigen = gsl_matrix_view_array_with_tda(c,width*height, existing_images*number_copies, existing_images*number_copies);

	w = gsl_eigen_symmv_alloc((number_copies)*existing_images);
	mean = calloc(width*height, sizeof(precision));

//  input_matrix = fopen("T.txt", "r");
	double temp;
	for(i = 0; i < number_copies; i++) {
		for(j = 0; j < existing_images; j++) {
			sprintf(full_path, "%s%d.ppm", file_path, j+1);
			testimage = ppm_image_constructor(full_path);
			grayscale(testimage);
			
			/*  copy image into the first matrix    */
			for(k = 0; k < width*height; k++) {
//              a[k + (i+1)*j*width*height] = testimage->pixels[k].r;
				gsl_matrix_set (&A.matrix, k, (i+1)*j, testimage->pixels[k].r);
				mean[k] += testimage->pixels[k].r;
			}
			
			ppm_image_destructor(testimage, 1);
		}
	}
	
	for(i = 0; i < width*height; i++) {
		mean[i] /= existing_images*number_copies;
	}
	
	/*  subtract the mean from the elements of matrix a */
	for(i = 0; i < number_copies*existing_images; i++) {
		for(j = 0; j < width*height; j++) {
			gsl_matrix_set (&A.matrix, j, i, gsl_matrix_get(&A.matrix, j,i) - mean[j]); //  testimage->pixels[k].r);
//          a[j + i*width*height] -= mean[j];
		}
	}
	printf("Mean subtracted, images database normalized...\n");
	fflush(stdout);

	/*  find covariance matrix "b"  */
	gsl_blas_dgemm (CblasTrans, CblasNoTrans,1.0, &A.matrix, &A.matrix, 0.0, &B.matrix);

	/*  find eigenvector and eigenvalues (doesn't give the same values as matlab or cula :-( )  */
	gsl_eigen_symmv (&B.matrix, &eigen_val_view.vector, &eigen_vec_view.matrix, w); //  &eigen_vec_view.matrix

	/*  you wouldn't believe how important normalized sorting is, but it's pretty important */
	gsl_eigen_symmv_sort (&eigen_val_view.vector, &eigen_vec_view.matrix, GSL_EIGEN_SORT_ABS_ASC);
	
	/*  project project!    */
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, &A.matrix, &eigen_vec_view.matrix, 0.0, &final_eigen.matrix);
	
	for(i=0;i<10;i++)
		printf("%le\n",c[i]);
		
	sprintf(full_path, "../database/eigen_%d.txt", number_copies*existing_images);
	eigen_file = fopen(full_path, "wb");
	i = number_copies*existing_images;
	fwrite(&i, sizeof(int), 1, eigen_file);
	i = number_copies*existing_images;
	fwrite(&i, sizeof(int), 1, eigen_file); /*  this variable is used if we reduced the eigen_vector    */
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
	eigen_file = fopen("results.txt", "a");
	sprintf(full_path, "%d Images at %lf (ms)\n", number_copies*existing_images, elapsedTime);
	fprintf(eigen_file, full_path);
	fclose(eigen_file);
}
