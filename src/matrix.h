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
#include "ppm.h"

typedef double precision_t;

typedef struct {
	precision_t *data;
	int rows;
	int cols;
} matrix_t;

#define elem(M, i, j) (M)->data[(j) * (M)->rows + (i)]

// constructor, destructor functions
matrix_t * m_initialize (int rows, int cols);
matrix_t * m_identity (int rows);
matrix_t * m_zeros (int rows, int cols);
matrix_t * m_copy (matrix_t *M);
matrix_t * m_copy_columns (matrix_t *M, int begin, int end);
void m_free (matrix_t *M);

// I/O functions
void m_fprint (FILE *stream, matrix_t *M);
void m_fwrite (FILE *stream, matrix_t *M);
matrix_t * m_fscan (FILE *stream);
matrix_t * m_fread (FILE *stream);
void m_ppm_read (matrix_t *M, int col, ppm_t *image);
void m_ppm_write (matrix_t *M, int col, ppm_t *image);

// getter functions
matrix_t * m_covariance (matrix_t *M);
precision_t m_dist_COS (matrix_t *A, int i, matrix_t *B, int j);
precision_t m_dist_L1 (matrix_t *A, int i, matrix_t *B, int j);
precision_t m_dist_L2 (matrix_t *A, int i, matrix_t *B, int j);
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t *M_eval, matrix_t *M_evec);
matrix_t * m_inverse (matrix_t *M);
matrix_t * m_mean_column (matrix_t *M);
matrix_t * m_product (matrix_t *A, matrix_t *B);
matrix_t * m_sqrtm (matrix_t *M);
matrix_t * m_transpose (matrix_t *M);

// mutator functions
void m_add (matrix_t *A, matrix_t *B);
void m_subtract (matrix_t *A, matrix_t *B);
void m_elem_mult (matrix_t *M, precision_t c);
void m_subtract_columns (matrix_t *M, matrix_t *a);

// TODO: functions to review
void m_flipCols (matrix_t *M);
void m_normalize (matrix_t *M);
matrix_t * m_reshape (matrix_t *M, int newNumRows, int newNumCols);
matrix_t * m_reorder_columns (matrix_t *M, matrix_t *V);

#endif
