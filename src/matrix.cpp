/**
 * @file matrix.cpp
 *
 * Implementation of the matrix library.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__NVCC__)
	#include <cuda_runtime.h>
	#include "magma_v2.h"
#else
	#include <cblas.h>
	#include <lapacke.h>
#endif

#include "logger.h"
#include "math_helper.h"
#include "matrix.h"

const precision_t EPSILON = 1e-16;

/**
 * Allocate memory on the GPU.
 *
 * @param size
 * @return pointer to memory
 */
void * gpu_malloc(size_t size)
{
	void *ptr = NULL;

#ifdef __NVCC__
	int stat = magma_malloc(&ptr, size);
	assert(stat == MAGMA_SUCCESS);
#endif

	return ptr;
}

/**
 * Free memory on the GPU.
 *
 * @param ptr
 */
void gpu_free(void *ptr)
{
#ifdef __NVCC__
	int stat = magma_free(ptr);
	assert(stat == MAGMA_SUCCESS);
#endif
}

/**
 * Allocate a cuBLAS matrix.
 *
 * @param M
 */
void gpu_malloc_matrix(matrix_t *M)
{
#ifdef __NVCC__
	M->data_gpu = (precision_t *)gpu_malloc(M->rows * M->cols * sizeof(precision_t));
#endif
}

/**
 * Get a MAGMA queue.
 *
 * @return MAGMA queue
 */
#ifdef __NVCC__
magma_queue_t magma_queue()
{
	static int init = 1;
	static int device = 0;
	static magma_queue_t queue;

	if ( init == 1 ) {
		magma_queue_create(device, &queue);

		init = 0;
	}

	return queue;
}
#endif

/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new matrix
 */
matrix_t * m_initialize (const char *name, int rows, int cols)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->name = name;
	M->rows = rows;
	M->cols = cols;
	M->data = (precision_t *)malloc(rows * cols * sizeof(precision_t));

	gpu_malloc_matrix(M);

	return M;
}

/**
 * Construct a matrix with arbitrary data.
 *
 * @param rows
 * @param cols
 * @param data
 * @return pointer to matrix
 */
matrix_t * m_initialize_data (const char *name, int rows, int cols, precision_t *data)
{
	matrix_t *M = m_initialize(name, rows, cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = data[i * cols + j];
		}
	}

	m_gpu_write(M);

	return M;
}

/**
 * Construct an identity matrix.
 *
 * @param rows
 * @return pointer to a new identity matrix
 */
matrix_t * m_identity (const char *name, int rows)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- eye(%d)\n",
		name, rows, rows,
		rows);

	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->name = name;
	M->rows = rows;
	M->cols = rows;
	M->data = (precision_t *)calloc(rows * rows, sizeof(precision_t));

	int i;
	for ( i = 0; i < rows; i++ ) {
		elem(M, i, i) = 1;
	}

	gpu_malloc_matrix(M);
	m_gpu_write(M);

	return M;
}

/**
 * Construct a matrix of all ones.
 *
 * @param rows
 * @param cols
 * @return pointer to a new ones matrix
 */
matrix_t * m_ones (const char *name, int rows, int cols)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- ones(%d, %d)\n",
		name, rows, cols,
		rows, cols);

	matrix_t *M = m_initialize(name, rows, cols);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			elem(M, i, j) = 1;
		}
	}

	m_gpu_write(M);

	return M;
}

/**
 * Construct a matrix of normally-distributed random numbers.
 *
 * @param rows
 * @param cols
 * @return pointer to a new random matrix
 */
matrix_t * m_random (const char *name, int rows, int cols)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- randn(%d, %d)\n",
		name, rows, cols,
		rows, cols);

	matrix_t *M = m_initialize(name, rows, cols);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			elem(M, i, j) = rand_normal(0, 1);
		}
	}

	m_gpu_write(M);

	return M;
}

/**
 * Construct a zero matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new zero matrix
 */
matrix_t * m_zeros (const char *name, int rows, int cols)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- zeros(%d, %d)\n",
		name, rows, cols,
		rows, cols);

	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->name = name;
	M->rows = rows;
	M->cols = cols;
	M->data = (precision_t *)calloc(rows * cols, sizeof(precision_t));

	gpu_malloc_matrix(M);
	m_gpu_write(M);

	return M;
}

/**
 * Copy a matrix.
 *
 * @param M
 * @return pointer to copy of M
 */
matrix_t * m_copy (const char *name, matrix_t *M)
{
	return m_copy_columns(name, M, 0, M->cols);
}

/**
 * Copy a range of columns in a matrix.
 *
 * @param M
 * @param i
 * @param j
 * @return pointer to copy of columns [i, j) of M
 */
matrix_t * m_copy_columns (const char *name, matrix_t *M, int i, int j)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s(:, %d:%d) [%d,%d]\n",
		name, M->rows, j - i,
		M->name, i + 1, j, M->rows, j - i);

	assert(0 <= i && i < j && j <= M->cols);

	matrix_t *C = m_initialize(name, M->rows, j - i);

	memcpy(C->data, &elem(M, 0, i), C->rows * C->cols * sizeof(precision_t));

	m_gpu_write(C);

	return C;
}

/**
 * Copy a range of rows in a matrix.
 *
 * @param M
 * @param i
 * @param j
 * @return pointer to copy of rows [i, j) of M
 */
matrix_t * m_copy_rows (const char *name, matrix_t *M, int i, int j)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s(%d:%d, :) [%d,%d]\n",
		name, j - i, M->cols,
		M->name, i + 1, j, j - i, M->cols);

	assert(0 <= i && i < j && j <= M->rows);

	matrix_t *C = m_initialize(name, j - i, M->cols);

	int k;
	for ( k = 0; k < M->cols; k++ ) {
		memcpy(&elem(C, 0, k), &elem(M, i, k), (j - i) * sizeof(precision_t));
	}

	m_gpu_write(C);

	return C;
}

/**
 * Deconstruct a matrix.
 *
 * @param M
 */
void m_free (matrix_t *M)
{
	free(M->data);
	gpu_free(M->data_gpu);
	free(M);
}

/**
 * Write a matrix in text format to a stream.
 *
 * @param stream
 * @param M
 */
void m_fprint (FILE *stream, matrix_t *M)
{
	fprintf(stream, "%s [%d, %d]\n", M->name, M->rows, M->cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			fprintf(stream, M_ELEM_FPRINT " ", elem(M, i, j));
		}
		fprintf(stream, "\n");
	}
}

/**
 * Write a matrix in binary format to a stream.
 *
 * @param stream
 * @param M
 */
void m_fwrite (FILE *stream, matrix_t *M)
{
	fwrite(&M->rows, sizeof(int), 1, stream);
	fwrite(&M->cols, sizeof(int), 1, stream);
	fwrite(M->data, sizeof(precision_t), M->rows * M->cols, stream);
}

/**
 * Read a matrix in text format from a stream.
 *
 * @param stream
 * @return pointer to new matrix
 */
matrix_t * m_fscan (FILE *stream)
{
	int rows, cols;
	fscanf(stream, "%d %d", &rows, &cols);

	matrix_t *M = m_initialize("", rows, cols);
	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			fscanf(stream, M_ELEM_FSCAN, &(elem(M, i, j)));
		}
	}

	return M;
}

/**
 * Read a matrix in binary format from a stream.
 *
 * @param stream
 * @return pointer to new matrix
 */
matrix_t * m_fread (FILE *stream)
{
	int rows, cols;
	fread(&rows, sizeof(int), 1, stream);
	fread(&cols, sizeof(int), 1, stream);

	matrix_t *M = m_initialize("", rows, cols);
	fread(M->data, sizeof(precision_t), M->rows * M->cols, stream);

	return M;
}

/**
 * Copy matrix data from device memory to host memory.
 *
 * @param M
 */
void m_gpu_read (matrix_t *M)
{
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_getmatrix(M->rows, M->cols, sizeof(precision_t),
		M->data_gpu, M->rows,
		M->data, M->rows,
		queue);
#endif
}

/**
 * Copy matrix data from host memory to device memory.
 *
 * @param M
 */
void m_gpu_write (matrix_t *M)
{
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_setmatrix(M->rows, M->cols, sizeof(precision_t),
		M->data, M->rows,
		M->data_gpu, M->rows,
		queue);
#endif
}

/**
 * Read a column vector from an image.
 *
 * @param M
 * @param i
 * @param image
 */
void m_image_read (matrix_t *M, int i, image_t *image)
{
	assert(M->rows == image->channels * image->height * image->width);

	int j;
	for ( j = 0; j < M->rows; j++ ) {
		elem(M, j, i) = (precision_t) image->pixels[j];
	}
}

/**
 * Write a column of a matrix to an image.
 *
 * @param M
 * @param i
 * @param image
 */
void m_image_write (matrix_t *M, int i, image_t *image)
{
	assert(M->rows == image->channels * image->height * image->width);

	int j;
	for ( j = 0; j < M->rows; j++ ) {
		image->pixels[j] = (unsigned char) elem(M, j, i);
	}
}

/**
 * Compute the diagonal matrix of a vector.
 *
 * @param v
 * @return pointer to diagonal matrix of v
 */
matrix_t * m_diagonalize (const char *name, matrix_t *v)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- diag(%s [%d,%d])\n",
		name, max(v->rows, v->cols), max(v->rows, v->cols),
		v->name, v->rows, v->cols);

	assert(v->rows == 1 || v->cols == 1);

	int n = (v->rows == 1)
		? v->cols
		: v->rows;
	matrix_t *D = m_zeros(name, n, n);

	int i;
	for ( i = 0; i < n; i++ ) {
		elem(D, i, i) = v->data[i];
	}

	m_gpu_write(D);

	return D;
}

/**
 * Compute the COS distance between two column vectors.
 *
 * Cosine similarity is the cosine of the angle between x and y:
 * S_cos(x, y) = x * y / (||x|| * ||y||)
 *
 * Since S_cos is on [-1, 1], we transform S_cos to be on [0, 2]:
 * d_cos(x, y) = 1 - S_cos(x, y)
 *
 * @param A
 * @param i
 * @param B
 * @param j
 * @return COS distance between A_i and B_j
 */
precision_t m_dist_COS (matrix_t *A, int i, matrix_t *B, int j)
{
	log(LL_DEBUG, "debug: d = COS(%s(:, %d) [%d,%d], %s(:, %d) [%d,%d])\n",
		A->name, i + 1, A->rows, 1,
		B->name, j + 1, B->rows, 1);

	assert(A->rows == B->rows);
	assert(0 <= i && i < A->cols && 0 <= j && j < B->cols);

	// compute x * y
	precision_t x_dot_y = 0;

	int k;
	for ( k = 0; k < A->rows; k++ ) {
		x_dot_y += elem(A, k, i) * elem(B, k, j);
	}

	// compute ||x|| and ||y||
	precision_t abs_x = 0;
	precision_t abs_y = 0;

	for ( k = 0; k < A->rows; k++ ) {
		abs_x += elem(A, k, i) * elem(A, k, i);
		abs_y += elem(B, k, j) * elem(B, k, j);
	}

	// compute similarity
	precision_t similarity = x_dot_y / sqrtf(abs_x * abs_y);

	// compute scaled distance
	return 1 - similarity;
}

/**
 * Compute the L1 distance between two column vectors.
 *
 * L1 is the Taxicab distance:
 * d_L1(x, y) = |x - y|
 *
 * @param A
 * @param i
 * @param B
 * @param j
 * @return L1 distance between A_i and B_j
 */
precision_t m_dist_L1 (matrix_t *A, int i, matrix_t *B, int j)
{
	log(LL_DEBUG, "debug: d = L1(%s(:, %d) [%d,%d], %s(:, %d) [%d,%d])\n",
		A->name, i + 1, A->rows, 1,
		B->name, j + 1, B->rows, 1);

	assert(A->rows == B->rows);
	assert(0 <= i && i < A->cols && 0 <= j && j < B->cols);

	precision_t dist = 0;

	int k;
	for ( k = 0; k < A->rows; k++ ) {
		dist += fabsf(elem(A, k, i) - elem(B, k, j));
	}

	return dist;
}

/**
 * Compute the L2 distance between two column vectors.
 *
 * L2 is the Euclidean distance:
 * d_L2(x, y) = ||x - y||
 *
 * @param A
 * @param i
 * @param B
 * @param j
 * @return L2 distance between A_i and B_j
 */
precision_t m_dist_L2 (matrix_t *A, int i, matrix_t *B, int j)
{
	log(LL_DEBUG, "debug: d = L2(%s(:, %d) [%d,%d], %s(:, %d) [%d,%d])\n",
		A->name, i + 1, A->rows, 1,
		B->name, j + 1, B->rows, 1);

	assert(A->rows == B->rows);
	assert(0 <= i && i < A->cols && 0 <= j && j < B->cols);

	precision_t dist = 0;

	int k;
	for ( k = 0; k < A->rows; k++ ) {
		precision_t diff = elem(A, k, i) - elem(B, k, j);
		dist += diff * diff;
	}

	dist = sqrtf(dist);

	return dist;
}

/**
 * Compute the eigenvalues and eigenvectors of a symmetric matrix.
 *
 * The eigenvalues are returned as a diagonal matrix, and the
 * eigenvectors are returned as column vectors. The i-th
 * eigenvalue corresponds to the i-th column vector. The eigenvalues
 * are returned in ascending order.
 *
 * @param M
 * @param n1
 * @param p_V
 * @param p_D
 */
void m_eigen (const char *V_name, const char *D_name, matrix_t *M, int n1, matrix_t **p_V, matrix_t **p_D)
{
	log(LL_DEBUG, "debug: %s [%d,%d], %s [%d,%d] <- eig(%s [%d,%d], %d)\n",
		V_name, M->rows, n1,
		D_name, n1, n1,
		M->name, M->rows, M->cols, n1);

	assert(M->rows == M->cols);

	matrix_t *V_temp1 = m_copy(V_name, M);
	matrix_t *D_temp1 = m_initialize(D_name, M->cols, 1);

	// solve A * x = lambda * x
#ifdef __NVCC__
	int n = M->cols;
	int nb = magma_get_ssytrd_nb(n);

	int ldwa = n;
	int lwork = max(2*n + n*nb, 1 + 6*n + 2*n*n);
	int liwork = 3 + 5*n;
	precision_t *wA = (precision_t *)malloc(ldwa * n * sizeof(precision_t));
	precision_t *work = (precision_t *)malloc(lwork * sizeof(precision_t));
	int *iwork = (int *)malloc(liwork * sizeof(int));
	int info;

	magma_ssyevd_gpu(MagmaVec, MagmaUpper,
		n, V_temp1->data_gpu, M->rows,  // input matrix (eigenvectors)
		D_temp1->data,                  // eigenvalues
		wA, ldwa,                       // workspace
		work, lwork,
		iwork, liwork,
		&info);
	assert(info == 0);

	free(wA);
	free(work);
	free(iwork);

	m_gpu_read(V_temp1);
#else
	int info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U',
		M->cols, V_temp1->data, M->rows,  // input matrix (eigenvectors)
		D_temp1->data);                   // eigenvalues
	assert(info == 0);
#endif

	// take only positive eigenvalues
	int i = 0;
	while ( i < D_temp1->rows && elem(D_temp1, i, 0) < EPSILON ) {
		i++;
	}

	// take only the n1 largest eigenvalues
	i = max(i, D_temp1->rows - n1);

	matrix_t *V = m_copy_columns(V_name, V_temp1, i, V_temp1->cols);
	matrix_t *D_temp2 = m_copy_rows(D_name, D_temp1, i, D_temp1->rows);

	// diagonalize eigenvalues
	matrix_t *D = m_diagonalize(D_name, D_temp2);

	// cleanup
	m_free(V_temp1);
	m_free(D_temp1);
	m_free(D_temp2);

	// save outputs
	*p_V = V;
	*p_D = D;
}

/**
 * Compute the inverse of a square matrix.
 *
 * @param M
 * @return pointer to new matrix equal to M^-1
 */
matrix_t * m_inverse (const char *name, matrix_t *M)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- inv(%s [%d,%d])\n",
		name, M->rows, M->cols,
		M->name, M->rows, M->cols);

	assert(M->rows == M->cols);

	matrix_t *M_inv = m_copy(name, M);

#ifdef __NVCC__
	int n = M->cols;
	int nb = magma_get_sgetri_nb(n);
	int lwork = n * nb;
	int *ipiv = (int *)malloc(n * sizeof(int));
	precision_t *dwork = (precision_t *)gpu_malloc(lwork * sizeof(precision_t));
	int info;

	magma_sgetrf_gpu(M->rows, n, M_inv->data_gpu, M->rows,
		ipiv, &info);
	assert(info == 0);

	magma_sgetri_gpu(n, M_inv->data_gpu, M->rows,
		ipiv, dwork, lwork, &info);
	assert(info == 0);

	free(ipiv);
	gpu_free(dwork);

	m_gpu_read(M_inv);
#else
	int *ipiv = (int *)malloc(M->cols * sizeof(int));

	int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR,
		M->rows, M->cols, M_inv->data, M->rows,
		ipiv);
	assert(info == 0);

	info = LAPACKE_sgetri(LAPACK_COL_MAJOR,
		M->cols, M_inv->data, M->rows,
		ipiv);
	assert(info == 0);

	free(ipiv);
#endif

	return M_inv;
}

/**
 * Get the mean column of a matrix.
 *
 * @param M
 * @return pointer to mean column vector
 */
matrix_t * m_mean_column (const char *name, matrix_t *M)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 2)\n",
		name, M->rows, 1,
		M->name, M->rows, M->cols);

	matrix_t *a = m_zeros(name, M->rows, 1);

	int i, j;
	for ( i = 0; i < M->cols; i++ ) {
		for ( j = 0; j < M->rows; j++ ) {
			elem(a, j, 0) += elem(M, j, i);
		}
	}
	m_gpu_write(a);

	m_elem_mult(a, 1.0f / M->cols);

	return a;
}

/**
 * Get the mean row of a matrix.
 *
 * @param M
 * @return pointer to mean row vector
 */
matrix_t * m_mean_row (const char *name, matrix_t *M)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 1)\n",
		name, 1, M->cols,
		M->name, M->rows, M->cols);

	matrix_t *a = m_zeros(name, 1, M->cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(a, 0, j) += elem(M, i, j);
		}
	}
	m_gpu_write(a);

	m_elem_mult(a, 1.0f / M->rows);

	return a;
}

/**
 * Compute the 2-norm of a vector.
 *
 * @param v
 * @return 2-norm of v
 */
precision_t m_norm (matrix_t *v)
{
	log(LL_DEBUG, "debug: n = norm(%s [%d,%d])\n",
		v->name, v->rows, v->cols);

	assert(v->rows == 1 || v->cols == 1);

	int N = (v->rows == 1)
		? v->cols
		: v->rows;
	int incX = 1;

	precision_t norm;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	norm = magma_snrm2(N, v->data_gpu, incX, queue);
#else
	norm = cblas_snrm2(N, v->data, incX);
#endif

	return norm;
}

/**
 * Get the product of two matrices.
 *
 * @param A
 * @param B
 * @param transA
 * @param transB
 * @return pointer to new matrix equal to A * B
 */
matrix_t * m_product (const char *name, matrix_t *A, matrix_t *B, bool transA, bool transB)
{
	int M = transA ? A->cols : A->rows;
	int K = transA ? A->rows : A->cols;
	int K2 = transB ? B->cols : B->rows;
	int N = transB ? B->rows : B->cols;

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s%s [%d,%d] * %s%s [%d,%d]\n",
		name, M, N,
		A->name, transA ? "'" : "", M, K,
		B->name, transB ? "'" : "", K2, N);

	assert(K == K2);

	matrix_t *C = m_zeros(name, M, N);

	precision_t alpha = 1;
	precision_t beta = 0;

	// C := alpha * A * B + beta * C
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();
	magma_trans_t TransA = transA ? MagmaTrans : MagmaNoTrans;
	magma_trans_t TransB = transB ? MagmaTrans : MagmaNoTrans;

	magma_sgemm(TransA, TransB,
		M, N, K,
		alpha, A->data_gpu, A->rows, B->data_gpu, B->rows,
		beta, C->data_gpu, C->rows,
		queue);

	m_gpu_read(C);
#else
	CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
	CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;

	cblas_sgemm(CblasColMajor, TransA, TransB,
		M, N, K,
		alpha, A->data, A->rows, B->data, B->rows,
		beta, C->data, C->rows);
#endif

	return C;
}

/**
 * Compute the principal square root of a symmetric matrix M. That
 * is, compute X such that X * X = M and X is the unique square root
 * for which every eigenvalue has non-negative real part.
 *
 * @param M
 * @return pointer to square root matrix of M
 */
matrix_t * m_sqrtm (const char *name, matrix_t *M)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- sqrtm(%s [%d,%d])\n",
		name, M->rows, M->cols,
		M->name, M->rows, M->cols);

	assert(M->rows == M->cols);

	// compute [V, D] = eig(M)
	matrix_t *V;
	matrix_t *D;

	m_eigen("V", "D", M, M->rows, &V, &D);

	// compute X = V * sqrt(D) * V'
	m_elem_apply(D, sqrtf);

	matrix_t *B = m_product("B", V, D);
	matrix_t *X = m_product(name, B, V, false, true);

	// cleanup
	m_free(B);
	m_free(V);
	m_free(D);

	return X;
}

/**
 * Compute the sum of the elements of a vector.
 *
 * @param v
 * @return sum of elements of v
 */
precision_t m_sum (matrix_t *v)
{
	log(LL_DEBUG, "debug: s = sum(%s [%d,%d])\n",
		v->name, v->rows, v->cols);

	assert(v->rows == 1 || v->cols == 1);

	int N = (v->rows == 1)
		? v->cols
		: v->rows;
	int incX = 1;

	precision_t sum = 0.0f;

	int i;
	for ( i = 0; i < N; i++ ) {
		sum += v->data[i];
	}

	return sum;
}

/**
 * Get the transpose of a matrix.
 *
 * NOTE: This function should not be necessary since
 * most transposes should be handled by m_product().
 *
 * @param M
 * @return pointer to new matrix M'
 */
matrix_t * m_transpose (const char *name, matrix_t *M)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- transpose(%s [%d,%d])\n",
		name, M->cols, M->rows,
		M->name, M->rows, M->cols);

	matrix_t *T = m_initialize(name, M->cols, M->rows);

	int i, j;
	for ( i = 0; i < T->rows; i++ ) {
		for ( j = 0; j < T->cols; j++ ) {
			elem(T, i, j) = elem(M, j, i);
		}
	}

	m_gpu_write(T);

	return T;
}

/**
 * Add a matrix to another matrix.
 *
 * @param A
 * @param B
 */
void m_add (matrix_t *A, matrix_t *B)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] + %s [%d,%d]\n",
		A->name, A->rows, A->cols,
		A->name, A->rows, A->cols,
		B->name, B->rows, B->cols);

	assert(A->rows == B->rows && A->cols == B->cols);

	int N = A->rows * A->cols;
	precision_t alpha = 1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(N, alpha, B->data_gpu, incX, A->data_gpu, incY, queue);

	m_gpu_read(A);
#else
	cblas_saxpy(N, alpha, B->data, incX, A->data, incY);
#endif
}

/**
 * Assign a column of a matrix.
 *
 * @param A  pointer to matrix
 * @param i  lhs column index
 * @param B  pointer to matrix
 * @param j  rhs column index
 */
void m_assign_column (matrix_t * A, int i, matrix_t * B, int j)
{
	log(LL_DEBUG, "debug: %s(:, %d) [%d,%d] <- %s(:, %d) [%d,%d]\n",
		A->name, i + 1, A->rows, 1,
		B->name, j + 1, B->rows, 1);

	assert(A->rows == B->rows);
	assert(0 <= i && i < A->cols);
	assert(0 <= j && j < B->cols);

	memcpy(&elem(A, 0, i), B->data, B->rows * sizeof(precision_t));

	m_gpu_write(A);
}

/**
 * Assign a row of a matrix.
 *
 * @param A  pointer to matrix
 * @param i  lhs row index
 * @param B  pointer to matrix
 * @param j  rhs row index
 */
void m_assign_row (matrix_t * A, int i, matrix_t * B, int j)
{
	log(LL_DEBUG, "debug: %s(%d, :) [%d,%d] <- %s(%d, :) [%d,%d]\n",
		A->name, i + 1, 1, A->cols,
		B->name, j + 1, 1, B->cols);

	assert(A->cols == B->cols);
	assert(0 <= i && i < A->rows);
	assert(0 <= j && j < B->rows);

	int k;
	for ( k = 0; k < A->cols; k++ ) {
		elem(A, i, k) = elem(B, j, k);
	}

	m_gpu_write(A);
}

/**
 * Apply a function to each element of a matrix.
 *
 * @param M
 * @param f
 */
void m_elem_apply (matrix_t * M, elem_func_t f)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- f(%s [%d,%d])\n",
		M->name, M->rows, M->cols,
		M->name, M->rows, M->cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = f(elem(M, i, j));
		}
	}

	m_gpu_write(M);
}

/**
 * Multiply a matrix by a scalar.
 *
 * @param M
 * @param c
 */
void m_elem_mult (matrix_t *M, precision_t c)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %g * %s [%d,%d]\n",
		M->name, M->rows, M->cols,
		c, M->name, M->rows, M->cols);

	int N = M->rows * M->cols;
	int incX = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_sscal(N, c, M->data_gpu, incX, queue);

	m_gpu_read(M);
#else
	cblas_sscal(N, c, M->data, incX);
#endif
}

/**
 * Subtract a matrix from another matrix.
 *
 * @param A
 * @param B
 */
void m_subtract (matrix_t *A, matrix_t *B)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d]\n",
		A->name, A->rows, A->cols,
		A->name, A->rows, A->cols,
		B->name, B->rows, B->cols);

	assert(A->rows == B->rows && A->cols == B->cols);

	int N = A->rows * A->cols;
	precision_t alpha = -1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(N, alpha, B->data_gpu, incX, A->data_gpu, incY, queue);

	m_gpu_read(A);
#else
	cblas_saxpy(N, alpha, B->data, incX, A->data, incY);
#endif
}

/**
 * Subtract a column vector from each column in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - a * 1_N'
 *
 * @param M  pointer to matrix
 * @param a  pointer to column vector
 */
void m_subtract_columns (matrix_t *M, matrix_t *a)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]\n",
		M->name, M->rows, M->cols,
		M->name, M->rows, M->cols,
		a->name, a->rows, a->cols,
		"1_N'", 1, M->cols);

	assert(M->rows == a->rows && a->cols == 1);

	int i, j;
	for ( i = 0; i < M->cols; i++ ) {
		for ( j = 0; j < M->rows; j++ ) {
			elem(M, j, i) -= elem(a, j, 0);
		}
	}
	m_gpu_write(M);
}

/**
 * Subtract a row vector from each row in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - 1_N * a
 *
 * @param M  pointer to matrix
 * @param a  pointer to row vector
 */
void m_subtract_rows (matrix_t *M, matrix_t *a)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]\n",
		M->name, M->rows, M->cols,
		M->name, M->rows, M->cols,
		"1_N", M->rows, 1,
		a->name, a->rows, a->cols);

	assert(M->cols == a->cols && a->rows == 1);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) -= elem(a, 0, j);
		}
	}
	m_gpu_write(M);
}
