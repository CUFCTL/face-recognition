/**
 * @file matrix.cpp
 *
 * Implementation of the matrix library.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include "logger.h"
#include "math_utils.h"
#include "matrix.h"

#if defined(__NVCC__)
	#include <cuda_runtime.h>
	#include "magma_v2.h"
#else
	#include <cblas.h>
	#include <lapacke.h>
#endif

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
	this->_name = name;
	this->_rows = rows;
	this->_cols = cols;
	this->_data_cpu = new precision_t[rows * cols];
	this->_data_gpu = gpu_malloc_matrix(rows, cols);
	this->_transposed = false;
	this->T = new Matrix();

	// initialize transpose
	this->T->_name = this->_name;
	this->T->_rows = rows;
	this->T->_cols = cols;
	this->T->_data_cpu = this->_data_cpu;
	this->T->_data_gpu = this->_data_gpu;
	this->T->_transposed = true;
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
	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			ELEM(*this, i, j) = data[i * cols + j];
		}
	}

	this->gpu_write();
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
	: Matrix(name, M._rows, j - i)
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s(:, %d:%d) [%d,%d]",
		this->_name, this->_rows, this->_cols,
		M._name, i + 1, j, M._rows, j - i);

	assert(0 <= i && i < j && j <= M._cols);

	memcpy(this->_data_cpu, &ELEM(M, 0, i), this->_rows * this->_cols * sizeof(precision_t));

	this->gpu_write();
}

/**
 * Copy a matrix.
 *
 * @param name
 * @param M
 */
Matrix::Matrix(const char *name, const Matrix& M)
	: Matrix(name, M, 0, M._cols)
{
}

/**
 * Copy-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(const Matrix& M)
	: Matrix(M._name, M, 0, M._cols)
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
	this->_name = "";
	this->_rows = 0;
	this->_cols = 0;
	this->_data_cpu = nullptr;
	this->_data_gpu = nullptr;
	this->_transposed = false;
	this->T = nullptr;
}

/**
 * Destruct a matrix.
 */
Matrix::~Matrix()
{
	if ( !this->_transposed ) {
		delete[] this->_data_cpu;
		gpu_free(this->_data_gpu);

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
	log(LL_DEBUG, "debug: %s [%d,%d] <- eye(%d)",
		name, rows, rows,
		rows);

	Matrix M(name, rows, rows);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < rows; j++ ) {
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- ones(%d, %d)",
		name, rows, cols,
		rows, cols);

	Matrix M(name, rows, cols);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- randn(%d, %d)",
		name, rows, cols,
		rows, cols);

	static std::default_random_engine generator;
	static std::normal_distribution<precision_t> distribution(0, 1);

	Matrix M(name, rows, cols);

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			ELEM(M, i, j) = distribution(generator);
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

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < cols; j++ ) {
			ELEM(M, i, j) = 0;
		}
	}

	M.gpu_write();

	return M;
}

/**
 * Print a matrix.
 *
 * @param os
 */
void Matrix::print(std::ostream& os) const
{
	os << this->_name << " [" << this->_rows << ", " << this->_cols << "]\n";

	for ( int i = 0; i < this->_rows; i++ ) {
		for ( int j = 0; j < this->_cols; j++ ) {
			os << std::right << std::setw(10) << std::setprecision(4) << this->elem(i, j);
		}
		os << "\n";
	}
}

/**
 * Save a matrix to a file.
 *
 * @param file
 */
void Matrix::save(std::ofstream& file) const
{
	file.write(reinterpret_cast<const char *>(&this->_rows), sizeof(int));
	file.write(reinterpret_cast<const char *>(&this->_cols), sizeof(int));
	file.write(reinterpret_cast<const char *>(this->_data_cpu), this->_rows * this->_cols * sizeof(precision_t));
}

/**
 * Load a matrix from a file.
 *
 * @param file
 */
void Matrix::load(std::ifstream& file)
{
	if ( this->_rows * this->_cols != 0 ) {
		log(LL_ERROR, "error: cannot load into non-empty matrix");
		exit(1);
	}

	int rows, cols;
	file.read(reinterpret_cast<char *>(&rows), sizeof(int));
	file.read(reinterpret_cast<char *>(&cols), sizeof(int));

	*this = Matrix("", rows, cols);
	file.read(reinterpret_cast<char *>(this->_data_cpu), this->_rows * this->_cols * sizeof(precision_t));
}

/**
 * Copy matrix data from device memory to host memory.
 */
void Matrix::gpu_read()
{
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_getmatrix(this->_rows, this->_cols, sizeof(precision_t),
		this->_data_gpu, this->_rows,
		this->_data_cpu, this->_rows,
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

	magma_setmatrix(this->_rows, this->_cols, sizeof(precision_t),
		this->_data_cpu, this->_rows,
		this->_data_gpu, this->_rows,
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
	assert(this->_rows == image.channels() * image.height() * image.width());

	for ( int j = 0; j < this->_rows; j++ ) {
		ELEM(*this, j, i) = (precision_t) image.elem(j);
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
	assert(this->_rows == image.channels() * image.height() * image.width());

	for ( int j = 0; j < this->_rows; j++ ) {
		image.elem(j) = (unsigned char) ELEM(*this, j, i);
	}
}

/**
 * Compute the determinant of a matrix using LU decomposition.
 *
 *   det(A) = det(P * L * U)
 */
precision_t Matrix::determinant() const
{
	int m = this->_rows;
	int n = this->_cols;
	Matrix U = *this;
	int lda = m;
	int *ipiv = new int[min(m, n)];

	// compute LU decomposition
#ifdef __NVCC__
	int info;

	magma_sgetrf_gpu(
		m, n, U._data_gpu, lda,
		ipiv,
		&info);
	assert(info == 0);

	U.gpu_read();
#else
	int info = LAPACKE_sgetrf_work(
		LAPACK_COL_MAJOR,
		m, n, U._data_cpu, lda,
		ipiv);
	assert(info == 0);
#endif

	// compute det(A) = det(P * L * U) = 1^S * det(U)
	precision_t det = 1;
	for ( int i = 0; i < min(m, n); i++ ) {
		if ( i + 1 != ipiv[i] ) {
			det *= -1;
		}
	}

	for ( int i = 0; i < min(m, n); i++ ) {
		det *= U.elem(i, i);
	}

	// cleanup
	delete[] ipiv;

	return det;
}

/**
 * Compute the diagonal matrix of a vector.
 *
 * @param name
 */
Matrix Matrix::diagonalize(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- diag(%s [%d,%d])",
		name, max(this->_rows, this->_cols), max(this->_rows, this->_cols),
		this->_name, this->_rows, this->_cols);

	assert(this->_rows == 1 || this->_cols == 1);

	int n = (this->_rows == 1)
		? this->_cols
		: this->_rows;
	Matrix D = Matrix::zeros(name, n, n);

	for ( int i = 0; i < n; i++ ) {
		ELEM(D, i, i) = this->_data_cpu[i];
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
	log(LL_DEBUG, "debug: %s [%d,%d], %s [%d,%d] <- eig(%s [%d,%d], %d)",
		V_name, this->_rows, n1,
		D_name, n1, n1,
		this->_name, this->_rows, this->_cols, n1);

	assert(this->_rows == this->_cols);

	V = Matrix(V_name, *this);
	D = Matrix(D_name, 1, this->_cols);

	// compute eigenvalues and eigenvectors
	int n = this->_cols;
	int lda = this->_rows;

#ifdef __NVCC__
	int nb = magma_get_ssytrd_nb(n);
	int ldwa = n;
	precision_t *wA = new precision_t[ldwa * n];
	int lwork = max(2*n + n*nb, 1 + 6*n + 2*n*n);
	precision_t *work = new precision_t[lwork];
	int liwork = 3 + 5*n;
	int *iwork = new int[liwork];
	int info;

	magma_ssyevd_gpu(MagmaVec, MagmaUpper,
		n, V._data_gpu, lda,
		D._data_cpu,
		wA, ldwa,
		work, lwork,
		iwork, liwork,
		&info);
	assert(info == 0);

	delete[] wA;
	delete[] work;
	delete[] iwork;

	V.gpu_read();
#else
	int lwork = 3 * n;
	precision_t *work = new precision_t[lwork];

	int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U',
		n, V._data_cpu, lda,
		D._data_cpu,
		work, lwork);
	assert(info == 0);

	delete[] work;
#endif

	// take only positive eigenvalues
	int i = 0;
	while ( i < D._cols && ELEM(D, 0, i) < EPSILON ) {
		i++;
	}

	// take only the n1 largest eigenvalues
	i = max(i, D._cols - n1);

	V = V(i, V._cols);
	D = D(i, D._cols).diagonalize(D_name);
}

/**
 * Compute the inverse of a square matrix.
 *
 * @param name
 */
Matrix Matrix::inverse(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- inv(%s [%d,%d])",
		name, this->_rows, this->_cols,
		this->_name, this->_rows, this->_cols);

	assert(this->_rows == this->_cols);

	Matrix M_inv(name, *this);

	int m = this->_rows;
	int n = this->_cols;
	int lda = this->_rows;

#ifdef __NVCC__
	int nb = magma_get_sgetri_nb(n);
	int *ipiv = new int[n];
	int lwork = n * nb;
	precision_t *dwork = (precision_t *)gpu_malloc(lwork * sizeof(precision_t));
	int info;

	magma_sgetrf_gpu(m, n, M_inv._data_gpu, lda,
		ipiv, &info);
	assert(info == 0);

	magma_sgetri_gpu(n, M_inv._data_gpu, lda,
		ipiv, dwork, lwork, &info);
	assert(info == 0);

	delete[] ipiv;
	gpu_free(dwork);

	M_inv.gpu_read();
#else
	int *ipiv = new int[n];
	int lwork = n;
	precision_t *work = new precision_t[lwork];

	int info = LAPACKE_sgetrf_work(LAPACK_COL_MAJOR,
		m, n, M_inv._data_cpu, lda,
		ipiv);
	assert(info == 0);

	info = LAPACKE_sgetri_work(LAPACK_COL_MAJOR,
		n, M_inv._data_cpu, lda,
		ipiv, work, lwork);
	assert(info == 0);

	delete[] ipiv;
	delete[] work;
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 2)",
		name, this->_rows, 1,
		this->_name, this->_rows, this->_cols);

	Matrix a = Matrix::zeros(name, this->_rows, 1);

	for ( int i = 0; i < this->_cols; i++ ) {
		for ( int j = 0; j < this->_rows; j++ ) {
			ELEM(a, j, 0) += ELEM(*this, j, i);
		}
	}
	a.gpu_write();

	a /= this->_cols;

	return a;
}

/**
 * Compute the mean row of a matrix.
 *
 * @param name
 */
Matrix Matrix::mean_row(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- mean(%s [%d,%d], 1)",
		name, 1, this->_cols,
		this->_name, this->_rows, this->_cols);

	Matrix a = Matrix::zeros(name, 1, this->_cols);

	for ( int i = 0; i < this->_rows; i++ ) {
		for ( int j = 0; j < this->_cols; j++ ) {
			ELEM(a, 0, j) += ELEM(*this, i, j);
		}
	}
	a.gpu_write();

	a /= this->_rows;

	return a;
}

/**
 * Compute the 2-norm of a vector.
 */
precision_t Matrix::norm() const
{
	log(LL_DEBUG, "debug: n = norm(%s [%d,%d])",
		this->_name, this->_rows, this->_cols);

	assert(this->_rows == 1 || this->_cols == 1);

	int n = (this->_rows == 1)
		? this->_cols
		: this->_rows;
	int incX = 1;

	precision_t norm;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	norm = magma_snrm2(n, this->_data_gpu, incX, queue);
#else
	norm = cblas_snrm2(n, this->_data_cpu, incX);
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

	int m = A._transposed ? A._cols : A._rows;
	int k1 = A._transposed ? A._rows : A._cols;
	int k2 = B._transposed ? B._cols : B._rows;
	int n = B._transposed ? B._rows : B._cols;

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s%s [%d,%d] * %s%s [%d,%d]",
		name, m, n,
		A._name, A._transposed ? "'" : "", m, k1,
		B._name, B._transposed ? "'" : "", k2, n);

	assert(k1 == k2);

	Matrix C = Matrix::zeros(name, m, n);

	precision_t alpha = 1.0f;
	precision_t beta = 0.0f;

	// C := alpha * A * B + beta * C
#ifdef __NVCC__
	magma_queue_t queue = magma_queue();
	magma_trans_t TransA = A._transposed ? MagmaTrans : MagmaNoTrans;
	magma_trans_t TransB = B._transposed ? MagmaTrans : MagmaNoTrans;

	magma_sgemm(TransA, TransB,
		m, n, k1,
		alpha, A._data_gpu, A._rows, B._data_gpu, B._rows,
		beta, C._data_gpu, C._rows,
		queue);

	C.gpu_read();
#else
	CBLAS_TRANSPOSE TransA = A._transposed ? CblasTrans : CblasNoTrans;
	CBLAS_TRANSPOSE TransB = B._transposed ? CblasTrans : CblasNoTrans;

	cblas_sgemm(CblasColMajor, TransA, TransB,
		m, n, k1,
		alpha, A._data_cpu, A._rows, B._data_cpu, B._rows,
		beta, C._data_cpu, C._rows);
#endif

	return C;
}

/**
 * Compute the sum of the elements of a vector.
 */
precision_t Matrix::sum() const
{
	log(LL_DEBUG, "debug: s = sum(%s [%d,%d])",
		this->_name, this->_rows, this->_cols);

	assert(this->_rows == 1 || this->_cols == 1);

	int n = (this->_rows == 1)
		? this->_cols
		: this->_rows;
	precision_t sum = 0.0f;

	for ( int i = 0; i < n; i++ ) {
		sum += this->_data_cpu[i];
	}

	return sum;
}

/**
 * Compute the economy-size singular value decomposition
 * of a matrix:
 *
 *   A = U * S * V'
 *
 * @param U
 * @param S
 * @param V
 */
void Matrix::svd(Matrix& U, Matrix& S, Matrix& V) const
{
	log(LL_DEBUG, "debug: U, S, V <- svd(%s [%d,%d])",
		this->_name, this->_rows, this->_cols);

	int m = this->_rows;
	int n = this->_cols;
	int lda = m;
	int ldu = m;
	int ldvt = min(m, n);

	U = Matrix("U", ldu, min(m, n));
	S = Matrix("S", 1, min(m, n));
	Matrix VT = Matrix("VT", ldvt, n);

#ifdef __NVCC__
	Matrix wA = *this;
	int nb = magma_get_sgesvd_nb(m, n);
	int lwork = 2 * min(m, n) + (max(m, n) + min(m, n)) * nb;
	precision_t *work = new precision_t[lwork];
	int info;

	magma_sgesvd(
		MagmaSomeVec, MagmaSomeVec,
		m, n, wA._data_cpu, lda,
		S._data_cpu,
		U._data_cpu, ldu,
		VT._data_cpu, ldvt,
		work, lwork,
		&info);
	assert(info == 0);

	delete[] work;
#else
	Matrix wA = *this;
	int lwork = 5 * min(m, n);
	precision_t *work = new precision_t[lwork];

	int info = LAPACKE_sgesvd_work(
		LAPACK_COL_MAJOR, 'S', 'S',
		m, n, wA._data_cpu, lda,
		S._data_cpu,
		U._data_cpu, ldu,
		VT._data_cpu, ldvt,
		work, lwork);
	assert(info == 0);

	delete[] work;
#endif

	S = S.diagonalize("S");
	V = VT.transpose("V");
}

/**
 * Compute the transpose of a matrix.
 *
 * @param name
 */
Matrix Matrix::transpose(const char *name) const
{
	log(LL_DEBUG, "debug: %s [%d,%d] <- transpose(%s [%d,%d])",
		name, this->_cols, this->_rows,
		this->_name, this->_rows, this->_cols);

	Matrix T(name, this->_cols, this->_rows);

	for ( int i = 0; i < T._rows; i++ ) {
		for ( int j = 0; j < T._cols; j++ ) {
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

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] + %s [%d,%d]",
		A._name, A._rows, A._cols,
		A._name, A._rows, A._cols,
		B._name, B._rows, B._cols);

	assert(A._rows == B._rows && A._cols == B._cols);

	int n = A._rows * A._cols;
	precision_t alpha = 1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(n, alpha, B._data_gpu, incX, A._data_gpu, incY, queue);

	A.gpu_read();
#else
	cblas_saxpy(n, alpha, B._data_cpu, incX, A._data_cpu, incY);
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

	log(LL_DEBUG, "debug: %s(:, %d) [%d,%d] <- %s(:, %d) [%d,%d]",
		A._name, i + 1, A._rows, 1,
		B._name, j + 1, B._rows, 1);

	assert(A._rows == B._rows);
	assert(0 <= i && i < A._cols);
	assert(0 <= j && j < B._cols);

	memcpy(&ELEM(A, 0, i), B._data_cpu, B._rows * sizeof(precision_t));

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

	log(LL_DEBUG, "debug: %s(%d, :) [%d,%d] <- %s(%d, :) [%d,%d]",
		A._name, i + 1, 1, A._cols,
		B._name, j + 1, 1, B._cols);

	assert(A._cols == B._cols);
	assert(0 <= i && i < A._rows);
	assert(0 <= j && j < B._rows);

	for ( int k = 0; k < A._cols; k++ ) {
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- f(%s [%d,%d])",
		this->_name, this->_rows, this->_cols,
		this->_name, this->_rows, this->_cols);

	for ( int i = 0; i < this->_rows; i++ ) {
		for ( int j = 0; j < this->_cols; j++ ) {
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- %g * %s [%d,%d]",
		this->_name, this->_rows, this->_cols,
		c, this->_name, this->_rows, this->_cols);

	int n = this->_rows * this->_cols;
	int incX = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_sscal(n, c, this->_data_gpu, incX, queue);

	this->gpu_read();
#else
	cblas_sscal(n, c, this->_data_cpu, incX);
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

	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d]",
		A._name, A._rows, A._cols,
		A._name, A._rows, A._cols,
		B._name, B._rows, B._cols);

	assert(A._rows == B._rows && A._cols == B._cols);

	int n = A._rows * A._cols;
	precision_t alpha = -1.0f;
	int incX = 1;
	int incY = 1;

#ifdef __NVCC__
	magma_queue_t queue = magma_queue();

	magma_saxpy(n, alpha, B._data_gpu, incX, A._data_gpu, incY, queue);

	A.gpu_read();
#else
	cblas_saxpy(n, alpha, B._data_cpu, incX, A._data_cpu, incY);
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]",
		this->_name, this->_rows, this->_cols,
		this->_name, this->_rows, this->_cols,
		a._name, a._rows, a._cols,
		"1_N'", 1, this->_cols);

	assert(this->_rows == a._rows && a._cols == 1);

	for ( int i = 0; i < this->_cols; i++ ) {
		for ( int j = 0; j < this->_rows; j++ ) {
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
	log(LL_DEBUG, "debug: %s [%d,%d] <- %s [%d,%d] - %s [%d,%d] * %s [%d,%d]",
		this->_name, this->_rows, this->_cols,
		this->_name, this->_rows, this->_cols,
		"1_N", this->_rows, 1,
		a._name, a._rows, a._cols);

	assert(this->_cols == a._cols && a._rows == 1);

	for ( int i = 0; i < this->_rows; i++ ) {
		for ( int j = 0; j < this->_cols; j++ ) {
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
	std::swap(A._name, B._name);
	std::swap(A._rows, B._rows);
	std::swap(A._cols, B._cols);
	std::swap(A._data_cpu, B._data_cpu);
	std::swap(A._data_gpu, B._data_gpu);
	std::swap(A._transposed, B._transposed);
	std::swap(A.T, B.T);
}
