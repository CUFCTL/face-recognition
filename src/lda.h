/**
 * @file lda.h
 *
 * Interface definitions for the LDA layer.
 */
#ifndef LDA_H
#define LDA_H

#include "dataset.h"
#include "matrix.h"

typedef struct {
	int n1;
	int n2;
} lda_params_t;

matrix_t * LDA(lda_params_t *params, matrix_t *X, int c, data_entry_t *entries);

#endif
