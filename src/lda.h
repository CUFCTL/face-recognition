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

matrix_t ** m_copy_classes(matrix_t *X, const std::vector<data_entry_t>& y, int c);
matrix_t ** m_class_means(matrix_t **X_c, int c);
matrix_t * m_scatter_between(matrix_t **X_c, matrix_t **U, int c);
matrix_t * m_scatter_within(matrix_t **X_c, matrix_t **U, int c);

matrix_t * LDA(lda_params_t *params, matrix_t *X, const std::vector<data_entry_t>& y, int c);

#endif
