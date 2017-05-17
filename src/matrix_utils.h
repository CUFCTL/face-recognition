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

typedef precision_t (*dist_func_t)(const Matrix&, int, const Matrix&, int);

precision_t m_dist_COS(const Matrix& A, int i, const Matrix& B, int j);
precision_t m_dist_L1(const Matrix& A, int i, const Matrix& B, int j);
precision_t m_dist_L2(const Matrix& A, int i, const Matrix& B, int j);

std::vector<Matrix> m_copy_classes(const Matrix& X, const std::vector<data_entry_t>& y, int c);
std::vector<Matrix> m_class_means(const std::vector<Matrix>& X_c, int c);
std::vector<Matrix> m_class_scatters(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U, int c);
Matrix m_scatter_between(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U, int c);
Matrix m_scatter_within(const std::vector<Matrix>& X_c, const std::vector<Matrix>& U, int c);

#endif
