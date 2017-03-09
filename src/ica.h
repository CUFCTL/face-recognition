/**
 * @file ica.h
 *
 * Interface definitions for the ICA layer.
 */
#ifndef ICA_H
#define ICA_H

#include "matrix.h"

typedef struct {
	int n1;
	int n2;
	int max_iterations;
	precision_t epsilon;
	dist_func_t dist;
} ica_params_t;

matrix_t * ICA(ica_params_t *params, matrix_t *X);

#endif
