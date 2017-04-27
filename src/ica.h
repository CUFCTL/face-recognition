/**
 * @file ica.h
 *
 * Interface definitions for the ICA layer.
 */
#ifndef ICA_H
#define ICA_H

#include "matrix.h"

typedef enum {
	ICA_NONL_NONE,
	ICA_NONL_POW3,
	ICA_NONL_TANH,
	ICA_NONL_GAUSS
} ica_nonl_t;

typedef matrix_t * (*ica_nonl_func_t)(matrix_t *, matrix_t *);

typedef struct {
	int n1;
	int n2;
	ica_nonl_t nonl;
	const char *nonl_name;
	int max_iterations;
	precision_t epsilon;
} ica_params_t;

matrix_t * ICA(ica_params_t *params, matrix_t *X);

#endif
