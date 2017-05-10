/**
 * @file matrix_utils.h
 *
 * Library of helpful matrix functions.
 */
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include "dataset.h"
#include "matrix.h"

matrix_t ** m_copy_classes(matrix_t *X, const std::vector<data_entry_t>& y, int c);
matrix_t ** m_class_means(matrix_t **X_c, int c);
matrix_t ** m_class_scatters(matrix_t **X_c, matrix_t **U, int c);
matrix_t * m_scatter_between(matrix_t **X_c, matrix_t **U, int c);
matrix_t * m_scatter_within(matrix_t **X_c, matrix_t **U, int c);

#endif
