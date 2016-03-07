/*
 *	pca_host.cu
 *
 *	Description: This file contains functions used to implement PCA with CUDA
 *
 *      atomicFloatAdd
 *      MatrixFindDistances
 *      cudasafe
 *      match_image
 *      Recognition
 *
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"
#include "pca.h"
#include <device_functions.h>
 
/*
 *	atomicFloatAdd
 *	
 *	parameters: 
 *      pointer, type float = address
 *      value, type float   = val
 *
 *	returns: N/A
 *      implicitly updates the value denoted by the variable "address"
 *
 *	Description: This function adds the value denoted by the varible "val" to the data stored in 
 *  the memory location denoted by the variable "address." This function uses the built-in, atomic
 *  function "atomicCAS" (atomic compare and swap) to ensure that there are no hazards and that no
 *  pertinent writes to the memory location are overwritten.
 *
 *      THIS FUNCTION CALLS
 *
 *      THIS FUNCTION IS CALLED BY
 *          MatrixFindDistances (pca_host.cu)
 *          
 *
 */
__device__ inline void atomicFloatAdd(float *address, float val) {
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0) {
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
}

/*
 *	MatrixFindDistances
 *
 *	parameters: 
 *      pointer, type eigen_type    = matrix
 *      pointer, type eigen_type    = vector
 *      value, type integer         = sizeW
 *      value, type integer         = sizeH
 *      pointer, type integer       = recognized_index
 *
 *	returns: N/A
 *
 *	Dexcription: 
 *
 *      THIS FUNCTION CALLS
 *          atomicFloatAdd  (pca_host.cu)
 *
 *      THIS FUNCTION IS CALLED BY
 *          match_image (pca_host.cu)
 *
 */
__global__ void MatrixFindDistances(eigen_type *matrix, eigen_type *vector, int sizeW, int sizeH, 
    int *recognized_index) {
	
	extern __shared__ eigen_type sdata[];
//  __shared__ unsigned int min_index[256];
	unsigned int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y;
	unsigned int s; /* iterators/reducers */
	unsigned int tid = threadIdx.x;
	unsigned int i = y_idx + sizeH * x_idx;

	/* moved to shared memory */
	if(x_idx < sizeW && y_idx < sizeH && tid < sizeW) {
		sdata[tid] = matrix[i];
	} else {
		sdata[tid] = 0;
	}
	__syncthreads();

	if(x_idx<sizeW && y_idx<sizeH) { 
		sdata[tid] = (sdata[tid] - vector[x_idx])*(sdata[tid] - vector[x_idx]); //square the difference
	}
	__syncthreads();

	/* reduction code for adding up the columns and the placing the results in row 0 */
	for (s=(blockDim.x)/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	/*  write result for this block to global mem */
	if (tid == 0) {
		/*  only 1 thread at a time can add (this is done at most 4 or so times, so parallelism 
		    isn't so important) */
		atomicFloatAdd(&matrix[y_idx], sdata[0]);	
	}
	__syncthreads();
}

/*
 *	cudasafe
 *
 *	parameters: 
 *      value, type cudaError_t = error
 *      pointer, type character = message
 *
 *	returns: N/A
 *
 *	Description: This function takes the variable "error" and checks to see if it is equivalent to 
 *  the constant "cudaSuccess." If it is not then this function will print the message pointed to by
 *  the other variable "message." The first argument is usually a CUDA command whose return value is
 *  type cudaError_t.
 *
 *      THIS FUNCTION CALLS
 *
 *      THIS FUNCTION IS CALLED BY
 *          match_image (pca_host.cu)
 *
 */
void cudasafe( cudaError_t error, char* message) {
	if(error!=cudaSuccess) { 
		fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1);
	}
}

/*
 *	match_image
 *
 *	parameters: 
 *      double pointer, type eigen_type = database
 *      double pointer, type eigen_type = image
 *      value, type integer             = sizeW
 *      value, type integer             = sizeH
 *
 *	returns: value, type integer
 *
 *	Description: 
 *
 *      THIS FUNCTION CALLS
 *          cudasafe            (pca_host.cu)
 *          MatrixFindDistances (pca_host.cu)
 *          
 *      THIS FUNCTION IS CALLED BY
 *          Recognition (pca_host.cu)
 *
 */
int match_image(eigen_type **database, eigen_type **image, int sizeW, int sizeH) {
	eigen_type *database_d, *image_d; /* pointers to device memory; a.k.a. GPU */
	int *recognized_index_d; 
	int recognized_index_h = -1;
	int blocksize=16;

	/* prepare host result array */

	/* allocate arrays on device */
	cudasafe(cudaMalloc((void **)&database_d,sizeH*sizeW*sizeof(eigen_type)), "Failed to allocate the image database on the CUDA device!");
	cudasafe(cudaMalloc((void **)&image_d,sizeH*sizeof(eigen_type)), "Failed to allocate test image on the CUDA device!");
	cudasafe(cudaMalloc((void **)&recognized_index_d,sizeof(int)), "Failed to allocate the recognized index prior to algorithm!");

	/* making a row-optimized block size (it's really 256 columns long per block, but 1 row tall) */
	dim3 dimBlock( blocksize * blocksize, 1); 
	dim3 dimGrid( ceil(float(sizeW)/float(dimBlock.x)), ceil(float(sizeH)/float(dimBlock.y)));
	dim3 dimGrid2( ceil(float(sizeW)/float(dimBlock.x)), 1);

	/* make sure to grab the size for the shared memory */
	size_t shmsize = size_t(dimBlock.x * dimBlock.y * dimBlock.z) * sizeof(float);
	size_t shmsize2 = size_t(dimBlock.x * dimBlock.y * dimBlock.z) * sizeof(int) + shmsize;

	/* copy and run the code on the device */
	cudasafe(cudaMemcpy(database_d,(*database),sizeW*sizeH*sizeof(eigen_type),cudaMemcpyHostToDevice), "Failed to copy host->device for image database!");
	cudasafe(cudaMemcpy(image_d,(*image),sizeH*sizeof(eigen_type),cudaMemcpyHostToDevice), "Failed to copy host->device for test image!");
	MatrixFindDistances<<<dimGrid, dimBlock, shmsize>>>(database_d,image_d, sizeW, sizeH, recognized_index_d);
	recognized_index_h = cublasIsamin (sizeW, database_d, 1);

//  MatrixFindMinimum<<<dimGrid, dimBlock, shmsize2>>>(database_d, sizeW, distances_list, min_index, size_list);
	if (cudaPeekAtLastError() != cudaSuccess) {
    		printf("kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
	cudaThreadSynchronize();
	
	/* grab the final index from the CUDA device */
//  cudasafe(cudaMemcpy(&recognized_index_h, recognized_index_d, sizeof(int),cudaMemcpyDeviceToHost), "Failed to retrieve final recognized index (device->host)!");	
//  cudasafe(cudaMemcpy((*image), database_d, sizeH*sizeof(eigen_type),cudaMemcpyDeviceToHost), "Failed to retrieve final recognized index (device->host)!");		
//  for(i = 0; i<100; i++)
//	    printf("%le\n", (*image)[i]);

	cudaFree(database_d);
	cudaFree(image_d);
	
	return(recognized_index_h);
}


/*
 *	Recognition
 *
 *	parameters: 
 *      pointer, type character         = inputimage
 *      double pointer, type eigen_type = mean
 *      double pointer, type eigen_type = projectedimages
 *      double pointer, type eigen_type = eigenfacesT
 *      double pointer, type eigen_type = projectedtrainimages
 *      pointer, type long integer      = images
 *      pointer, type long integer      = imgsize
 *
 *	returns: N/A
 *
 *	Description: 
 *
 *      THIS FUNCTION CALLS
 *          match_image (pca_host.cu)
 *
 *      THIS FUNCTION IS CALLED BY
 *          main    (pca.cu)
 *
 */
void Recognition(char *inputimage, eigen_type **mean, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, eigen_type **projectedtrainimages, long int *images, 
    long int *imgsize) {
	
	char outputtext[100];
	long int i, j;
	int min_index;
	eigen_type 	*projectedtestimage;
	eigen_type	*testimage_normalized;
	PPMImage 	*testimage;

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
	for(i = 0; i < *images-1; i++) {
		projectedtestimage[i] = 0;
		for(j = 0; j < *imgsize; j++) {
			projectedtestimage[i] += (*eigenfacesT)[j*(*images-1) + i] * testimage_normalized[j];
		}
	}

	/* perform the matching in cuda */
	min_index = match_image(projectedtrainimages, &projectedtestimage, (*images - 1), *images) - 1;
	
	eigen_type difference, distance, smallest = 1e20;
	for(i = 0; i < *images; i++) {
		distance = 0;
		for(j = 0; j < (*images)-1; j++) {
			difference = (*projectedtrainimages)[j*(*images) + i] - projectedtestimage[j];
			distance += difference*difference;
		}
		
//      if(!PROFILING) printf("distance[%d] = %le\n====\n", i, distance);
		
		if(distance < smallest) {
			smallest = distance;
			min_index = i;
		}
	}
	
	sprintf(outputtext, "%s matches image index %d\n", inputimage, min_index);
	printf("\n%s\n", outputtext);
}

