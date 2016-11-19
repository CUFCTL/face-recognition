/**
 * @file matrix.h
 *
 * Interface definitions for the matrix library.
 *
 * NOTE: Unlike C, which stores static arrays in row-major
 * order, this library stores matrices in column-major order.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include "image.h"

typedef double precision_t;

typedef struct {
	precision_t *data;
	int rows;
	int cols;
#ifdef __NVCC__
	precision_t *data_dev;
#endif
} matrix_t;

typedef precision_t (*elem_func_t)(precision_t);

#define elem(M, i, j) (M)->data[(j) * (M)->rows + (i)]

// cuBLAS helper functions
void cublas_set_matrix(matrix_t *M);
void cublas_get_matrix(matrix_t *M);

// constructor, destructor functions
matrix_t * m_initialize (int rows, int cols);
matrix_t * m_identity (int rows);
matrix_t * m_ones (int rows, int cols);
matrix_t * m_random (int rows, int cols);
matrix_t * m_zeros (int rows, int cols);
matrix_t * m_copy (matrix_t *M);
matrix_t * m_copy_columns (matrix_t *M, int i, int j);
matrix_t * m_copy_rows (matrix_t *M, int i, int j);
void m_free (matrix_t *M);

// I/O functions
void m_fprint (FILE *stream, matrix_t *M);
void m_fwrite (FILE *stream, matrix_t *M);
matrix_t * m_fscan (FILE *stream);
matrix_t * m_fread (FILE *stream);
void m_image_read (matrix_t *M, int col, image_t *image);
void m_image_write (matrix_t *M, int col, image_t *image);

// getter functions
matrix_t * m_covariance (matrix_t *M);
matrix_t * m_diagonalize (matrix_t *v);
precision_t m_dist_COS (matrix_t *A, int i, matrix_t *B, int j);
precision_t m_dist_L1 (matrix_t *A, int i, matrix_t *B, int j);
precision_t m_dist_L2 (matrix_t *A, int i, matrix_t *B, int j);
void m_eigen (matrix_t *M, matrix_t **p_V, matrix_t **p_D);
void m_eigen2 (matrix_t *A, matrix_t *B, matrix_t **p_V, matrix_t **p_D);
matrix_t * m_inverse (matrix_t *M);
matrix_t * m_mean_column (matrix_t *M);
matrix_t * m_mean_row (matrix_t *M);
precision_t m_norm(matrix_t *v);
matrix_t * m_product (matrix_t *A, matrix_t *B);
matrix_t * m_sqrtm (matrix_t *M);
matrix_t * m_transpose (matrix_t *M);

// mutator functions
void m_add (matrix_t *A, matrix_t *B);
void m_assign_column (matrix_t * A, int i, matrix_t * B, int j);
void m_assign_row (matrix_t * A, int i, matrix_t * B, int j);
void m_elem_apply (matrix_t * M, elem_func_t f);
void m_elem_mult (matrix_t *M, precision_t c);
void m_shuffle_columns (matrix_t *M);
void m_subtract (matrix_t *A, matrix_t *B);
void m_subtract_columns (matrix_t *M, matrix_t *a);
void m_subtract_rows (matrix_t *M, matrix_t *a);

#endif
