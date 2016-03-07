/*==================================================================================================
 *	pca_host.cu
 *
 *  Edited by: William Halsey
 *  whalsey@g.clemson.edu
 *
 *  THIS FILE CONTAINS
 *      atomicFloatAdd
 *      k_grayscale_normalize
 *      k_project_image
 *      k_project_image_collect
 *      MatrixFindDistances
 *      cudasafe
 *      match_image
 *      Recognition
 *
 *	Description: 
 *
 *  Last edited: Jul. 18, 2013
 *  Edits:
 *      
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"
#include "pca.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

/*==================================================================================================
 *	atomicFloatAdd
 *
 *	parameters
 *	    single pointer, type float  = address
 *      variable, type float        = val
 *
 *  returns
 *      N/A
 *      This function manipulates the value held in "address."
 *
 *	Description: This function employs the built in CUDA function atomicCAS in order to atomically
 *      add the variable "val" to the value contained at "address."
 *  In this CUDA implementation of the PCA algorithm, this will allow only one thread at a time to
 *      modify the value at "address."
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      MatrixFindDistances (pca_host.cu)
 *
 */
__device__ inline void atomicFloatAdd(float *address, float val) {
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while((tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0) {
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
	
	return;
}


/*==================================================================================================
 *	k_grayscale_normalize
 *
 *	parameters
 *      single pointer, type Pixel      = device_image
 *      variable, type int              = sizeH
 *      variable, type int              = sizeW
 *      single pointer, type eigen_type = mean
 *      single pointer, type eigen_type = test_image_norm
 *
 *	returns
 *      N/A
 *      Implicitly returns a value through variable "test_image_norm."
 *
 *	Description: This function takes the inputted image "device_image" and subtract the average
 *      image, "mean," and store the resulting normalized image in "test_image_norm."
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      Recognition (pca_host.cu)
 *
 */
__global__ void k_grayscale_normalize(Pixel *device_image, int sizeH, int sizeW, eigen_type *mean,
    eigen_type *test_image_norm) {

	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y;
	unsigned int i = y_idx + sizeH * x_idx;
	
	/*  moved to shared memory  */
	if (x_idx < sizeW && y_idx < sizeH) {
	    /*  ((int)(.2989 * (device_image)[i].r + .5870 * (device_image)[i].g + .1140 * (device_image)[i].b)) - (mean)[i] + 1;   */
		(test_image_norm)[i] = device_image[i].r - mean[i] + 1;
	}
	
	return;
}


/*==================================================================================================
 *	k_project_image
 *
 *	parameters
 *      single pointer, type eigen_type = test_image_norm
 *      single pointer, type eigen_type = test_image_d2
 *      variable, type int              = img_size
 *      variable, type int              = num_faces
 *      single pointer, type eigen_type = eigenfacesT
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      Recognition (pca_host.cu)
 */
__global__ void k_project_image(eigen_type *test_image_norm, eigen_type *test_image_d2,
    int img_size, int num_faces, eigen_type *eigenfacesT) {

	extern __shared__ eigen_type sdata[];
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y;
	unsigned int s; /*  iterators/reducers  */
	unsigned int tid = threadIdx.x;
//  unsigned int i = y_idx + sizeH * x_idx;

    /*  moved to shared memory  */
	if (y_idx < num_faces && x_idx < img_size && tid < img_size)
		sdata[tid] = (eigenfacesT)[x_idx * (num_faces) + y_idx] * (test_image_norm)[x_idx];
	else
		sdata[tid] = 0;
//  __syncthreads();

	/*  reduction code for adding up the columns and the placing the results in row 0   */
	for (s = (blockDim.x) / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	/*  write result for this block to global mem   */
	if (tid == 0) {
		test_image_d2[y_idx + blockIdx.x*num_faces] = sdata[tid]; 
	}
	
	return;
}


/*==================================================================================================
 *	k_project_image_collect
 *
 *	parameters
 *      single pointer, type eigen_type = test_image_d2
 *      variable, type int              = image_size
 *      variable, type int              = width
 *      variable, type int              = final_width
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      Recognition (pca_host.cu)
 */
__global__ void k_project_image_collect(eigen_type *test_image_d2, int image_size, int width,
    int final_width) {

	extern __shared__ eigen_type sdata[];
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y;
	unsigned int s; /*  iterators/reducers  */
	unsigned int tid = threadIdx.x;
//  unsigned int i = y_idx + sizeH * x_idx;

	/*  moved to shared memory  */
	if(y_idx < image_size && x_idx < width && tid < width)
		sdata[tid] = test_image_d2[y_idx + x_idx * image_size];
	else
		sdata[tid] = 0;
//  __syncthreads();

	/*  reduction code for adding up the columns and the placing the results in row 0   */
	for (s=(blockDim.x) / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	/*  write result for this block to global mem   */
	if (tid == 0) {
		test_image_d2[0] = sdata[0]; 
	}
	
	return;
}
 

/*==================================================================================================
 *	MatrixFindDistances
 *
 *	parameters
 *      single pointer, type eigen_type = matrix
 *      single pointer, type eigen_type = vector
 *      variable, type int              = sizeW
 *      variable, type int              = sizeH
 *      single pointer, type int        = recognized_index
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *      atomicFloatAdd  (pca_host.cu)
 *
 *  THIS FUNCTION IS CALLED BY
 *      match_image (pca_host.cu)
 *
 */
__global__ void MatrixFindDistances(eigen_type *matrix, eigen_type *vector, int sizeW, int sizeH,
    int *recognized_index) {

	extern __shared__ eigen_type sdata[];
//  __shared__ unsigned int min_index[256];
	unsigned int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y;
	unsigned int s; /*  iterators/reducers  */
	unsigned int tid = threadIdx.x;
	unsigned int i = y_idx + sizeH * x_idx;

	/*  moved to shared memory  */
	if(x_idx < sizeW && y_idx < sizeH && tid < sizeW)
		sdata[tid] = matrix[i];
	else
		sdata[tid] = 0;
	__syncthreads();

	if(x_idx<sizeW && y_idx<sizeH) { 
		sdata[tid] = (sdata[tid] - vector[x_idx])*(sdata[tid] - vector[x_idx]); /*  square the difference   */
	}
	__syncthreads();

	/*  reduction code for adding up the columns and the placing the results in row 0   */
	for (s=(blockDim.x)/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	/*  write result for this block to global mem   */
	if (tid == 0) {
		/*  only 1 thread at a time can add (this is done at most 4 or so times, so parallelism isn't so important) */
		atomicFloatAdd(&matrix[y_idx], sdata[0]);	
	}
	
	return;
}


/*==================================================================================================
 *	cudasafe
 *
 *	parameters
 *      variable, type cudaError_t  = error
 *      single pointer, type char   = message
 *
 *	returns
 *      N/A
 *
 *	Description: This function is used to handle any errors that may occur with built in CUDA 
 *      function calls. The function first compares "error" with the CUDA defined constant
 *      "cudaSuccess." If they are equivalent nothing happens, but if they are not then "message"
 *      and "error" are both printed to stderr and the program is terminated.
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *      main        (pca.cu)
 *      Recognition (pca_host.cu)
 *
 */
void cudasafe(cudaError_t error, char* message) {
	if(error != cudaSuccess) {
	    fprintf(stderr,"ERROR: %s : %i\n",message,error);
	    exit(-1);
	}
	
	return;
}


/*==================================================================================================
 *	match_image
 *
 *	parameters
 *      double pointer, type eigen_type = database
 *      double pointer, type eigen_type = image
 *      variable, type int              = sizeW
 *      variable, type int              = sizeH
 *      double pointer, type eigen_type = database_d
 *      double pointer, type int        = recognized_index_d
 *
 *	returns
 *      variable, type int
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *      MatrixFindDistance  (pca_host.cu)
 *
 *  THIS FUNCTION IS CALLED BY
 *      Recognition (pca_host.cu)
 */
int match_image(eigen_type **database, eigen_type **image, int sizeW, int sizeH,
    eigen_type **database_d, int **recognized_index_d) {

	int recognized_index_h = -1;
	int blocksize=16;

	/*  making a row-optimized block size (it's really 256 columns long per block, but 1 row tall)  */
	dim3 dimBlock( blocksize * blocksize, 1); 
	dim3 dimGrid( ceil(float(sizeW)/float(dimBlock.x)), ceil(float(sizeH)/float(dimBlock.y)));

	/*  make sure to grab the size for the shared memory    */
	size_t shmsize = size_t(dimBlock.x * dimBlock.y * dimBlock.z) * sizeof(float);
//  size_t shmsize2 = size_t(dimBlock.x * dimBlock.y * dimBlock.z) * sizeof(int) + shmsize;

	/*  put the database in memory  */
//  cudasafe(cudaMemcpy((*database_d),(*database),sizeH * sizeW*sizeof(eigen_type),cudaMemcpyHostToDevice), "Failed to copy host->device for image database!");
//  cudasafe(cudaMemcpy((*image_d),(*image),sizeH*sizeof(eigen_type),cudaMemcpyHostToDevice), "Failed to copy host->device for test image!");
	
	MatrixFindDistances<<<dimGrid, dimBlock, shmsize>>>(*database_d,*image, sizeW, sizeH, *recognized_index_d);

	recognized_index_h = cublasIsamin (sizeW, (*database_d), 1);	

	if (cudaPeekAtLastError() != cudaSuccess) {
    		printf("kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	cudaThreadSynchronize();
	
	return(recognized_index_h);
}


/*==================================================================================================
 *	Recognition
 *
 *	paramters
 *      single pointer, type char       = inputimage
 *      double pointer, type eigen_type = mean_d
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = projectedtrainimages
 *      single pointer, type long int   = images
 *      single pointer, type long int   = imgsize
 *      single pointer, type long int   = facessize
 *      double pointer, type eigen_type = database_d
 *      double pointer, type eigen_type = image_d
 *      double pointer, type int        = recognized_index_d
 *      double pointer, type Pixel      = test_image_d
 *      double pointer, type eigen_type = test_image_d2
 *      double pointer, type eigen_type = test_image_norm
 *
 *	returns
 *      N/A
 *
 *	Description: 
 *
 *  THIS FUNCTION CALLS
 *      ppm_image_constructor   (ppm.cu)
 *      cudasafe                (pca_host.cu)
 *      k_grayscale_normalize   (pca_host.cu)
 *      k_project_image         (pca_host.cu)
 *      k_project_image_collect (pca_host.cu)
 *      match_image             (pca_host.cu)
 *      ppm_image_destructor    (ppm.cu)
 *
 *  THIS FUNCTION IS CALLED BY
 *      main    (pca.cu)
 *
 */
void Recognition(char *inputimage, eigen_type **mean_d, eigen_type **projectedimages,
    eigen_type **eigenfacesT, eigen_type **projectedtrainimages, long int *images,
    long int *imgsize, long int *facessize, eigen_type **database_d, eigen_type **image_d,
    int **recognized_index_d, Pixel** test_image_d, eigen_type **test_image_d2,
    eigen_type **test_image_norm) {
	
	char outputtext[30];
	int min_index;
	Pixel *junk = (Pixel *)malloc(sizeof(Pixel) * (100));;
	PPMImage 	*testimage;

	/*  timing variables    */
	cudaEvent_t start, stop;
	float elapsedTime;

	int blocksize=32;
	printf("%s\n", inputimage);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/*  read in test image  */
	testimage = ppm_image_constructor(inputimage);
//  grayscale(testimage);
	
	/*  making a row-optimized block size (it's really 256 columns long per block, but 1 row tall)  */
	dim3 dimBlock( blocksize * blocksize, 1);
	dim3 dimGrid( ceil(float(testimage->width)/float(dimBlock.x)), ceil(float(testimage->height)/float(dimBlock.y)));

	dim3 dimGrid2( ceil(float(*imgsize)/float(dimBlock.x)), ceil(float(*facessize)/float(dimBlock.y)));
	size_t shmsize = size_t(dimBlock.x * dimBlock.y * dimBlock.z) * sizeof(float);
	
	/*  copy to device memory   */
	cudasafe(cudaMemcpy((*test_image_d),(testimage->pixels),sizeof(Pixel)*testimage->height*testimage->width,cudaMemcpyHostToDevice), "Failed to copy host->device for test image1!");
	
//  cudasafe(cudaMemcpy((*test_image_d),(testimage->pixels),sizeof(Pixel)*testimage->height*testimage->width,cudaMemcpyHostToDevice), "Failed to copy host->device for mean vector!");
	k_grayscale_normalize<<<dimGrid, dimBlock>>>(*test_image_d, testimage->height, testimage->width, *mean_d, *test_image_norm);

//  projectedtestimage = (eigen_type *)malloc(sizeof(eigen_type) * (*facessize));
//  cudasafe(cudaMemcpy((projectedtestimage),(*test_image_norm), sizeof(eigen_type)*20,cudaMemcpyDeviceToHost), "Failed to copy device->host for pixels!");
	
	k_project_image<<<dimGrid2, dimBlock, shmsize>>>(*test_image_norm,*test_image_d2, *imgsize, *facessize, *eigenfacesT);
	k_project_image_collect<<<dimGrid2, dimBlock, shmsize>>>(*test_image_d2, *facessize, (*imgsize/(blocksize*blocksize)), 0);
//  cudasafe(cudaMemcpy((projectedtestimage),(*test_image_d2), sizeof(eigen_type)*((*facessize)),cudaMemcpyDeviceToHost), "Failed to copy device->host for test image!");
	
//	for(i = 0; i<10; i++) {
//		printf("projectedtestimage: %le\n", projectedtestimage[i]);
//	}
	
    /*  project test image (ProjectedTestImage = eigenfacesT * NormalizedInputImage)    */
	
	/*  perform the matching in cuda    */
	min_index = match_image(projectedtrainimages, test_image_d2, (*facessize), *images, database_d, recognized_index_d) - 1;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\nOverall Speed:\t\t\t\t%lf (ms)\n", elapsedTime);
	
//	sprintf(outputtext, "%s matches image index %d.ppm\n", inputimage, min_index + 1);
//	printf("%s", outputtext);

//	free(projectedtestimage);
//	free(testimage_normalized);
	ppm_image_destructor(testimage, 1);

    return;
}

