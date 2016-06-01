#ifndef INCLUDE_MAIN
#define INCLUDE_MAIN

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


#include "ppm.h"

#define existing_images 200
//#define number_copies 1

void grayscale(PPMImage* img);

#endif
