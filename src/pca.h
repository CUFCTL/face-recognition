/**
 * @file pca.h
 *
 * Interface definitions for the PCA layer.
 */
#ifndef PCA_H
#define PCA_H

#include "matrix.h"

typedef struct {
	int n1;
} pca_params_t;

matrix_t * PCA(pca_params_t *params, matrix_t *X, matrix_t **p_D);

#endif
