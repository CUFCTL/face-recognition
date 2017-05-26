/**
 * @file matrix.cpp
 *
 * Implementation of the matrix library.
 */
#include <algorithm>
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
#include "math_utils.h"
#include "matrix.h"

const precision_t EPSILON = 1e-16;

/**
 * Allocate memory on the GPU.
 *
 * @param size
 */
void * gpu_malloc(size_t size)
{
	void *ptr = nullptr;

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
 * Allocate a matrix on the GPU.
 *
 * @param rows
 * @param cols
 */
precision_t * gpu_malloc_matrix(int rows, int cols)
{
	return (precision_t *)gpu_malloc(rows * cols * sizeof(precision_t));
}

/**
 * Get a MAGMA queue.
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
 * @param name
 * @param rows
 * @param cols
 */
Matrix::Matrix(const char *name, int rows, int cols)
{
	this->name = name;
	this->rows = rows;
	this->cols = cols;
	this->data = (precision_t *)malloc(rows * cols * sizeof(precision_t));
	this->data_gpu = gpu_malloc_matrix(rows, cols);
	this->transposed = false;
	this->T = new Matrix();

	// initialize transpose
	this->T->name = this->name;
	this->T->rows = rows;
	this->T->cols = cols;
	this->T->data = this->data;
	this->T->data_gpu = this->data_gpu;
	this->T->transposed = true;
	this->T->T = nullptr;
}

/**
 * Construct a matrix with arbitrary data.
 *
 * @param name
 * @param rows
 * @param cols
 * @param data
 */
Matrix::Matrix(const char *name, int rows, int cols, precision_t *data)
	: Matrix(name, rows, cols)
{
	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			ELEM(*this, i, j) = data[i * cols + j];
		}
	}

	this->gpu_write();
}

/**
 * Copy a matrix.
 *
 * @param name
 * @param M
 */
Matrix::Matrix(const char *name, const Matrix& M)
	: Matrix(name, M, 0, M.cols)
{
}

/**
 * Copy a range of columns in a matrix.
 *
 * @param name
 * @param M
 * @param i
 * @param j
 */
Matrix::Matrix(const char *name, const Matrix& M, int i, int j)
	: Matrix(name, M.rows, j - i)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s(:, %d:%d) [%d,%d]\n",
		this->name, this->rows, this->cols,
		M.name, i + 1, j, M.rows, j - i);

	assert(0 <= i && i < j && j <= M.cols);

	memcpy(this->data, &ELEM(M, 0, i), this->rows * this->cols * sizeof(precision_t));

	this->gpu_write();
}

/**
 * Copy-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(const Matrix& M)
	: Matrix(M.name, M, 0, M.cols)
{
}

/**
 * Move-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(Matrix&& M)
	: Matrix()
{
	swap(*this, M);
}

/**
 * Construct an empty matrix.
 */
Matrix::Matrix()
{
	this->name = "";
	this->rows = 0;
	this->cols = 0;
	this->data = nullptr;
	this->data_gpu = nullptr;
	this->transposed = false;
	this->T = nullptr;
}

/**
 * Destruct a matrix.
 */
Matrix::~Matrix()
{
	if ( !this->transposed ) {
		free(this->data);
		gpu_free(this->data_gpu);

		delete this->T;
	}
}

/**
 * Construct an identity matrix.
 *
 * @param name
 * @param rows
 */
Matrix Matrix::identity(const char *name, int rows)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- eye(%d)\n",
		name, rows, rows,
		rows);

	Matrix M(name, rows, rows);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < rows; j++ ) {
			ELEM(M, i, j) = (i == j);
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Construct a matrix of all ones.
 *
 * @param name
 * @param rows
 * @param cols
 */
Matrix Matrix::ones(const char *name, int rows, int cols)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- ones(%d, %d)\n",
		name, rows, cols,
		rows, cols);

	Matrix M(name, rows, cols);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			ELEM(M, i, j) = 1;
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Construct a matrix of normally-distributed random numbers.
 *
 * @param name
 * @param rows
 * @param cols
 */
Matrix Matrix::random(const char *name, int rows, int cols)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- randn(%d, %d)\n",
		name, rows, cols,
		rows, cols);

	Matrix M(name, rows, cols);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			ELEM(M, i, j) = rand_normal(0, 1);
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Construct a zero matrix.
 *
 * @param name
 * @param rows
 * @param cols
 */
Matrix Matrix::zeros(const char *name, int rows, int cols)
{
	Matrix M(name, rows, cols);

	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			ELEM(M, i, j) = 0;
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Print a matrix to a file.
 *
 * @param file
 */
void Matrix::print(FILE *file) const
{
	fprintf(file, "%s [%d, %d]\n", this->name, this->rows, this->cols);

	int i, j;
	for ( i = 0; i < this->rows; i++ ) {
		for ( j = 0; j < this->cols; j++ ) {
			fprintf(file, M_ELEM_FPRINT " ", ELEM(*this, i, j));
		}
		fprintf(file, "\n");
	}
}

/**
 * Save a matrix to a file.
 *
 * @param file
 */
void Matrix::save(FILE *file) const
{
	fwrite(&this->rows, sizeof(int), 1, file);
	fwrite(&this->cols, sizeof(int), 1, file);
	fwrite(this->data, sizeof(precision_t), this->rows * this->cols, file);
}

/**
 * Load a matrix from a file.
 *
 * @param file
 */
void Matrix::load(FILE *file)
{
	if ( this->rows * this->cols != 0 ) {
		log(LL_ERROR, "error: cannot load into non-empty matrix");
		exit(1);
	}

	int rows, cols;
	fread(&rows, sizeof(int), 1, file);
	fread(&cols, sizeof(int), 1, file);

	Matrix("", rows, cols);
	fread(this->data, sizeof(precision_t), this->rows * this->cols, file);
}

/**
 * Copy matrix data from device memory to host memory.
 */
void Matrix::gpu_read()
{
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_getmatrix(this->rows, this->cols, sizeof(precision_t),
		this->data_gpu, this->rows,
		this->data, this->rows,
		queue);
#endif
}

/**
 * Copy matrix data from host memory to device memory.
 */
void Matrix::gpu_write()
{
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_setmatrix(this->rows, this->cols, sizeof(precision_t),
		this->data, this->rows,
		this->data_gpu, this->rows,
		queue);
#endif
}

/**
 * Read a column vector from an image.
 *
 * @param i
 * @param image
 */
void Matrix::image_read(int i, const Image& image)
{
	assert(this->rows == image.channels * image.height * image.width);

	int j;
	for ( j = 0; j < this->rows; j++ ) {
		ELEM(*this, j, i) = (precision_t) image.pixels[j];
	}
}

/**
 * Write a column vector to an image.
 *
 * @param i
 * @param image
 */
void Matrix::image_write(int i, Image& image)
{
	assert(this->rows == image.channels * image.height * image.width);

	int j;
	for ( j = 0; j < this->rows; j++ ) {
		image.pixels[j] = (unsigned char) ELEM(*this, j, i);
	}
}

/**
 * Determine the index of the first element
 * in a vector that is the maximum value.
 */
int Matrix::argmax() const
{
	assert(this->rows == 1 || this->cols == 1);

	int n = (this->rows == 1)
		? this->cols
		: this->rows;

	int index = 0;
	precision_t max = this->data[0];

	int i;
	for ( i = 1; i < n; i++ ) {
		if ( max < this->data[i] ) {
			max = this->data[i];
			index = i;
		}
	}

	return index;
}

/**
 * Compute the diagonal matrix of a vector.
 *
 * @param name
 */
Matrix Matrix::diagonalize(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- diag(%s [%d,%d])\n",
		name, max(this->rows, this->cols), max(this->rows, this->cols),
		this->name, this->rows, this->cols);

	assert(this->rows == 1 || this->cols == 1);

	int n = (this->rows == 1)
		? this->cols
		: this->rows;
	Matrix D = Matrix::zeros(name, n, n);

	int i;
	for ( i = 0; i < n; i++ ) {
		ELEM(D, i, i) = this->data[i];
	}

	D.gpu_write();

	return D;
}

/**
 * Compute the eigenvalues and eigenvectors of a symmetric matrix.
 *
 * The eigenvalues are returned as a diagonal matrix, and the
 * eigenvectors are returned as column vectors. The i-th
 * eigenvalue corresponds to the i-th column vector. The eigenvalues
 * are returned in ascending order.
 *
 * @param V_name
 * @param D_name
 * @param n1
 * @param V
 * @param D
 */
void Matrix::eigen(const char *V_name, const char *D_name, int n1, Matrix& V, Matrix& D) const
{
	log(LL_DEBUG, "debug: %s [%d,%d], %s [%d,%d] <- eig(%s [%d,%d], %d)\n",
		V_name, this->rows, n1,
		D_name, n1, n1,
		this->name, this->rows, this->cols, n1);

	assert(this->rows == this->cols);

	Matrix V_temp1(V_name, *this);
	Matrix D_temp1(D_name, 1, this->cols);

	// compute eigenvalues and eigenvectors
	int n = this->cols;
	int lda = this->rows;

#ifdef __NVCC__
	int nb = magma_get_ssytrd_nb(n);
	int ldwa = n;
	precision_t *wA = (precision_t *)malloc(ldwa * n * sizeof(precision_t));
	int lwork = max(2*n + n*nb, 1 + 6*n + 2*n*n);
	precision_t *work = (precision_t *)malloc(lwork * sizeof(precision_t));
	int liwork = 3 + 5*n;
	int *iwork = (int *)malloc(liwork * sizeof(int));
	int info;

	magma_ssyevd_gpu(MagmaVec, MagmaUpper,
		n, V_temp1.data_gpu, lda,
		D_temp1.data,
		wA, ldwa,
		work, lwork,
		iwork, liwork,
		&info);
	assert(info == 0);

	free(wA);
	free(work);
	free(iwork);

	V_temp1.gpu_read();
#else
	int lwork = 3 * n;
	precision_t *work = (precision_t *)malloc(lwork * sizeof(precision_t));

	int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U',
		n, V_temp1.data, lda,
		D_temp1.data,
		work, lwork);
	assert(info == 0);

	free(work);
#endif

	// take only positive eigenvalues
	int i = 0;
	while ( i < D_temp1.cols && ELEM(D_temp1, 0, i) < EPSILON ) {
		i++;
	}

	// take only the n1 largest eigenvalues
	i = max(i, D_temp1.cols - n1);

	V = Matrix(V_name, V_temp1, i, V_temp1.cols);

	Matrix D_temp2 = Matrix(D_name, D_temp1, i, D_temp1.cols);
	D = D_temp2.diagonalize(D_name);
}

/**
 * Compute the inverse of a square matrix.
 *
 * @param name
 */
Matrix Matrix::inverse(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- inv(%s [%d,%d])\n",
		name, this->rows, this->cols,
		this->name, this->rows, this->cols);

	assert(this->rows == this->cols);

	Matrix M_inv(name, *this);

	int m = this->rows;
	int n = this->cols;
	int lda = this->rows;

#ifdef __NVCC__
	int nb = magma_get_sgetri_nb(n);
	int *ipiv = (int *)malloc(n * sizeof(int));
	int lwork = n * nb;
	precision_t *dwork = (precision_t *)gpu_malloc(lwork * sizeof(precision_t));
	int info;

	magma_sgetrf_gpu(m, n, M_inv.data_gpu, lda,
		ipiv, &info);
	assert(info == 0);

	magma_sgetri_gpu(n, M_inv.data_gpu, lda,
		ipiv, dwork, lwork, &info);
	assert(info == 0);

	free(ipiv);
	gpu_free(dwork);

	M_inv.gpu_read();
#else
	int *ipiv = (int *)malloc(n * sizeof(int));
	int lwork = n;
	precision_t *work = (precision_t *)malloc(lwork * sizeof(precision_t));

	int info = LAPACKE_sgetrf_work(LAPACK_COL_MAJOR,
		m, n, M_inv.data, lda,
		ipiv);
	assert(info == 0);

	info = LAPACKE_sgetri_work(LAPACK_COL_MAJOR,
		n, M_inv.data, lda,
		ipiv, work, lwork);
	assert(info == 0);

	free(ipiv);
	free(work);
#endif

	return M_inv;
}

/**
 * Compute the mean column of a matrix.
 *
 * @param name
 */
Matrix Matrix::mean_column(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 2)\n",
		name, this->rows, 1,
		this->name, this->rows, this->cols);

	Matrix a = Matrix::zeros(name, this->rows, 1);

	int i, j;
	for ( i = 0; i < this->cols; i++ ) {
		for ( j = 0; j < this->rows; j++ ) {
			ELEM(a, j, 0) += ELEM(*this, j, i);
		}
	}
	a.gpu_write();

	a.elem_mult(1.0f / this->cols);

	return a;
}

/**
 * Compute the mean row of a matrix.
 *
 * @param name
 */
Matrix Matrix::mean_row(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 1)\n",
		name, 1, this->cols,
		this->name, this->rows, this->cols);

	Matrix a = Matrix::zeros(name, 1, this->cols);

	int i, j;
	for ( i = 0; i < this->rows; i++ ) {
		for ( j = 0; j < this->cols; j++ ) {
			ELEM(a, 0, j) += ELEM(*this, i, j);
		}
	}
	a.gpu_write();

	a.elem_mult(1.0f / this->rows);

	return a;
}

/**
 * Compute the 2-norm of a vector.
 */
precision_t Matrix::norm() const
{
	log(LL_DEBUG, "debug: n = norm(%s [%d,%d])\n",
		this->name, this->rows, this->cols);

	assert(this->rows == 1 || this->cols == 1);

	int n = (this->rows == 1)
		? this->cols
		: this->rows;
	int incX = 1;

	precision_t norm;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	norm = magma_snrm2(n, this->data_gpu, incX, queue);
#else
	norm = cblas_snrm2(n, this->data, incX);
#endif

	return norm;
}

/**
 * Compute the product of two matrices.
 *
 * @param name
 * @param B
 */
Matrix Matrix::product(const char *name, const Matrix& B) const
{
	const Matrix& A = *this;

	int m = A.transposed ? A.cols : A.rows;
	int k1 = A.transposed ? A.rows : A.cols;
	int k2 = B.transposed ? B.cols : B.rows;
	int n = B.transposed ? B.rows : B.cols;

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s%s [%d,%d] * %s%s [%d,%d]\n",
		name, m, n,
		A.name, A.transposed ? "'" : "", m, k1,
		B.name, B.transposed ? "'" : "", k2, n);

	assert(k1 == k2);

	Matrix C = Matrix::zeros(name, m, n);

	precision_t alpha = 1.0f;
	precision_t beta = 0.0f;

	// C := alpha * A * B + beta * C
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();
	magma_trans_t TransA = A.transposed ? MagmaTrans : MagmaNoTrans;
	magma_trans_t TransB = B.transposed ? MagmaTrans : MagmaNoTrans;

	magma_sgemm(TransA, TransB,
		m, n, k1,
		alpha, A.data_gpu, A.rows, B.data_gpu, B.rows,
		beta, C.data_gpu, C.rows,
		queue);

	C.gpu_read();
#else
	CBLAS_TRANSPOSE TransA = A.transposed ? CblasTrans : CblasNoTrans;
	CBLAS_TRANSPOSE TransB = B.transposed ? CblasTrans : CblasNoTrans;

	cblas_sgemm(CblasColMajor, TransA, TransB,
		m, n, k1,
		alpha, A.data, A.rows, B.data, B.rows,
		beta, C.data, C.rows);
#endif

	return C;
}

/**
 * Compute the sum of the elements of a vector.
 */
precision_t Matrix::sum() const
{
	log(LL_DEBUG, "debug: s = sum(%s [%d,%d])\n",
		this->name, this->rows, this->cols);

	assert(this->rows == 1 || this->cols == 1);

	int n = (this->rows == 1)
		? this->cols
		: this->rows;
	precision_t sum = 0.0f;

	int i;
	for ( i = 0; i < n; i++ ) {
		sum += this->data[i];
	}

	return sum;
}

/**
 * Compute the transpose of a matrix.
 *
 * @param name
 */
Matrix Matrix::transpose(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- transpose(%s [%d,%d])\n",
		name, this->cols, this->rows,
		this->name, this->rows, this->cols);

	Matrix T(name, this->cols, this->rows);

	int i, j;
	for ( i = 0; i < T.rows; i++ ) {
		for ( j = 0; j < T.cols; j++ ) {
			ELEM(T, i, j) = ELEM(*this, j, i);
		}
	}

	T.gpu_write();

	return T;
}

/**
 * Add a matrix to another matrix.
 *
 * @param B
 */
void Matrix::add(const Matrix& B)
{
	Matrix& A = *this;

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] + %s [%d,%d]\n",
		A.name, A.rows, A.cols,
		A.name, A.rows, A.cols,
		B.name, B.rows, B.cols);

	assert(A.rows == B.rows && A.cols == B.cols);

	int n = A.rows * A.cols;
	precision_t alpha = 1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(n, alpha, B.data_gpu, incX, A.data_gpu, incY, queue);

	A.gpu_read();
#else
	cblas_saxpy(n, alpha, B.data, incX, A.data, incY);
#endif
}

/**
 * Assign a column of a matrix.
 *
 * @param i
 * @param B
 * @param j
 */
void Matrix::assign_column(int i, const Matrix& B, int j)
{
	Matrix& A = *this;

	log(LL_DEBUG, "debug: %s(:, %d) [%d,%d] <- %s(:, %d) [%d,%d]\n",
		A.name, i + 1, A.rows, 1,
		B.name, j + 1, B.rows, 1);

	assert(A.rows == B.rows);
	assert(0 <= i && i < A.cols);
	assert(0 <= j && j < B.cols);

	memcpy(&ELEM(A, 0, i), B.data, B.rows * sizeof(precision_t));

	A.gpu_write();
}

/**
 * Assign a row of a matrix.
 *
 * @param i
 * @param B
 * @param j
 */
void Matrix::assign_row(int i, const Matrix& B, int j)
{
	Matrix& A = *this;

	log(LL_DEBUG, "debug: %s(%d, :) [%d,%d] <- %s(%d, :) [%d,%d]\n",
		A.name, i + 1, 1, A.cols,
		B.name, j + 1, 1, B.cols);

	assert(A.cols == B.cols);
	assert(0 <= i && i < A.rows);
	assert(0 <= j && j < B.rows);

	int k;
	for ( k = 0; k < A.cols; k++ ) {
		ELEM(A, i, k) = ELEM(B, j, k);
	}

	A.gpu_write();
}

/**
 * Apply a function to each element of a matrix.
 *
 * @param f
 */
void Matrix::elem_apply(elem_func_t f)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- f(%s [%d,%d])\n",
		this->name, this->rows, this->cols,
		this->name, this->rows, this->cols);

	int i, j;
	for ( i = 0; i < this->rows; i++ ) {
		for ( j = 0; j < this->cols; j++ ) {
			ELEM(*this, i, j) = f(ELEM(*this, i, j));
		}
	}

	this->gpu_write();
}

/**
 * Multiply a matrix by a scalar.
 *
 * @param c
 */
void Matrix::elem_mult(precision_t c)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %g * %s [%d,%d]\n",
		this->name, this->rows, this->cols,
		c, this->name, this->rows, this->cols);

	int n = this->rows * this->cols;
	int incX = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_sscal(n, c, this->data_gpu, incX, queue);

	this->gpu_read();
#else
	cblas_sscal(n, c, this->data, incX);
#endif
}

/**
 * Subtract a matrix from another matrix.
 *
 * @param B
 */
void Matrix::subtract(const Matrix& B)
{
	Matrix& A = *this;

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d]\n",
		A.name, A.rows, A.cols,
		A.name, A.rows, A.cols,
		B.name, B.rows, B.cols);

	assert(A.rows == B.rows && A.cols == B.cols);

	int n = A.rows * A.cols;
	precision_t alpha = -1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(n, alpha, B.data_gpu, incX, A.data_gpu, incY, queue);

	A.gpu_read();
#else
	cblas_saxpy(n, alpha, B.data, incX, A.data, incY);
#endif
}

/**
 * Subtract a column vector from each column in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - a * 1_N'
 *
 * @param a
 */
void Matrix::subtract_columns(const Matrix& a)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]\n",
		this->name, this->rows, this->cols,
		this->name, this->rows, this->cols,
		a.name, a.rows, a.cols,
		"1_N'", 1, this->cols);

	assert(this->rows == a.rows && a.cols == 1);

	int i, j;
	for ( i = 0; i < this->cols; i++ ) {
		for ( j = 0; j < this->rows; j++ ) {
			ELEM(*this, j, i) -= ELEM(a, j, 0);
		}
	}
	this->gpu_write();
}

/**
 * Subtract a row vector from each row in a matrix.
 *
 * This function is equivalent to:
 *
 *   M = M - 1_N * a
 *
 * @param a
 */
void Matrix::subtract_rows(const Matrix& a)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]\n",
		this->name, this->rows, this->cols,
		this->name, this->rows, this->cols,
		"1_N", this->rows, 1,
		a.name, a.rows, a.cols);

	assert(this->cols == a.cols && a.rows == 1);

	int i, j;
	for ( i = 0; i < this->rows; i++ ) {
		for ( j = 0; j < this->cols; j++ ) {
			ELEM(*this, i, j) -= ELEM(a, 0, j);
		}
	}
	this->gpu_write();
}

/**
 * Swap function for Matrix.
 *
 * @param A
 * @param B
 */
void swap(Matrix& A, Matrix& B)
{
	std::swap(A.name, B.name);
	std::swap(A.rows, B.rows);
	std::swap(A.cols, B.cols);
	std::swap(A.data, B.data);
	std::swap(A.data_gpu, B.data_gpu);
	std::swap(A.transposed, B.transposed);
	std::swap(A.T, B.T);
}
