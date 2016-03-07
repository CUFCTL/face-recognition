/*
 *  pca.h
 *
 *  Description: 
 *
 */
 
#ifndef pca_header
#define pca_header

#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include "cublas.h"

//#include <cuda.h>

#define eigen_type float /* types of values for eigenfaces and projected images matrices */

#define DEBUG 0 /* 1 for printf debug statements */
#define PROFILING 0 /* 1 if testing recognition time */

extern void LoadTrainingDatabase(char *filename, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, eigen_type **mean, long int *images, long int *imgsize);

extern void Recognition(char *inputimage, eigen_type **mean, eigen_type **projectedimages,
    eigen_type **eigenfacesT, eigen_type **projectedtrainimages, long int *images, 
    long int *imgsize);

#endif

