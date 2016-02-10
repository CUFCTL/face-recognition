/*==================================================================================================
 *  main.h
 *  
 *  edited by: William Halsey
 *  whalsey@g.clmeson.edu
 *
 *  Description: 
 *
 *  Last edited: Jul. 15, 2013
 *  Edits: 
 *
 */
#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <string.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <cula.h>
#include <cula_blas.h>
#include <cula_device.h>
#include <cula_blas_device.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"

/*  types of values for eigenfaces and projected images matrices    */
#define eigen_type double

extern void LoadTrainingDatabase(char *filename, eigen_type **projectedimages, 
    eigen_type **eigenfacesT, eigen_type **mean, int *images, int *imgsize, int *num_faces);

#endif

