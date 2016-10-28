/**
 * @file matrix.c
 *
 * Implementation of the matrix library.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>
#include "matrix.h"

/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new matrix
 */
matrix_t * m_initialize (int rows, int cols)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->rows = rows;
	M->cols = cols;
	M->data = (precision_t *) malloc(rows * cols * sizeof(precision_t));

	return M;
}

/**
 * Construct an identity matrix.
 *
 * @param rows
 * @return pointer to a new identity matrix
 */
matrix_t * m_identity (int rows)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->rows = rows;
	M->cols = rows;
	M->data = (precision_t *) calloc(rows * rows, sizeof(precision_t));

	int i;
	for ( i = 0; i < rows; i++ ) {
		elem(M, i, i) = 1;
	}

	return M;
}

/**
 * Construct a matrix of all ones.
 *
 * @param rows
 * @param cols
 * @return pointer to a new ones matrix
 */
matrix_t * m_ones(int rows, int cols)
{
    matrix_t *M = m_initialize(rows, cols);

    int i, j;
    for ( i = 0; i < rows; i++ ) {
        for ( j = 0; j < cols; j++ ) {
            elem(M, i, j) = 1;
        }
    }

    return M;
}

/**
 * Generate a normally-distributed (mu, sigma) random number
 * using the Box-Muller transform.
 *
 * @param mu      mean
 * @param signma  standard deviation
 * @return normally-distributed random number
 */
precision_t rand_normal(precision_t mu, precision_t sigma)
{
	static int init = 1;
	static int generate = 0;
	static precision_t z0, z1;

	// provide a seed on the first call
	if ( init ) {
		srand48(1);
		init = 0;
	}

	// return z1 if z0 was returned in the previous call
	generate = !generate;
	if ( !generate ) {
		return z1 * sigma + mu;
	}

	// generate number pair (z0, z1), return z0
	precision_t u1 = drand48();
	precision_t u2 = drand48();

	z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

	return z0 * sigma + mu;
}

/**
 * Construct a matrix of normally-distributed random numbers.
 *
 * @param rows
 * @param cols
 * @return pointer to a new random matrix
 */
matrix_t * m_random (int rows, int cols)
{
    matrix_t *M = m_initialize(rows, cols);

    int i, j;
    for ( i = 0; i < rows; i++ ) {
        for ( j = 0; j < cols; j++ ) {
            elem(M, i, j) = rand_normal(0, 1);
        }
    }

    return M;
}

/**
 * Construct a zero matrix.
 *
 * @param rows
 * @param cols
 * @return pointer to a new zero matrix
 */
matrix_t * m_zeros (int rows, int cols)
{
	matrix_t *M = (matrix_t *)malloc(sizeof(matrix_t));
	M->rows = rows;
	M->cols = cols;
	M->data = (precision_t *) calloc(rows * cols, sizeof(precision_t));

	return M;
}

/**
 * Copy a matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to copy of M
 */
matrix_t * m_copy (matrix_t *M)
{
	return m_copy_columns(M, 0, M->cols);
}

/**
 * Copy a range of columns in a matrix.
 *
 * @param M      pointer to matrix
 * @param begin  begin index
 * @param end    end index
 * @return pointer to copy of columns [begin, end) of M
 */
matrix_t * m_copy_columns (matrix_t *M, int begin, int end)
{
	assert(0 <= begin && begin < end && end <= M->cols);

	matrix_t *C = m_initialize(M->rows, end - begin);

	memcpy(C->data, &elem(M, 0, begin), C->rows * C->cols * sizeof(precision_t));

	return C;
}

/**
 * Deconstruct a matrix.
 *
 * @param M  pointer to matrix
 */
void m_free (matrix_t *M)
{
	free(M->data);
	free(M);
}

/**
 * Write a matrix in text format to a stream.
 *
 * @param stream  pointer to file stream
 * @param M       pointer to matrix
 */
void m_fprint (FILE *stream, matrix_t *M)
{
	fprintf(stream, "%d %d\n", M->rows, M->cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			fprintf(stream, "% 8.4lf ", elem(M, i, j));
		}
		fprintf(stream, "\n");
	}
}

/**
 * Write a matrix in binary format to a stream.
 *
 * @param stream  pointer to file stream
 * @param M       pointer to matrix
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
 * @param stream  pointer to file stream
 * @return pointer to new matrix
 */
matrix_t * m_fscan (FILE *stream)
{
	int rows, cols;
	fscanf(stream, "%d %d", &rows, &cols);

	matrix_t *M = m_initialize(rows, cols);
	int i, j;
	for ( i = 0; i < rows; i++ ) {
		for ( j = 0; j < cols; j++ ) {
			fscanf(stream, "%lf", &(elem(M, i, j)));
		}
	}

	return M;
}

/**
 * Read a matrix in binary format from a stream.
 *
 * @param stream  pointer to file stream
 * @return pointer to new matrix
 */
matrix_t * m_fread (FILE *stream)
{
	int rows, cols;
	fread(&rows, sizeof(int), 1, stream);
	fread(&cols, sizeof(int), 1, stream);

	matrix_t *M = m_initialize(rows, cols);
	fread(M->data, sizeof(precision_t), M->rows * M->cols, stream);

	return M;
}

/**
 * Read a column vector from an image.
 *
 * @param M      pointer to matrix
 * @param col    column index
 * @param image  pointer to image
 */
void m_image_read (matrix_t *M, int col, image_t *image)
{
	assert(M->rows == image->channels * image->height * image->width);

	int i;
	for ( i = 0; i < M->rows; i++ ) {
		elem(M, i, col) = (precision_t) image->pixels[i];
	}
}

/**
 * Write a column of a matrix to an image.
 *
 * @param M      pointer to matrix
 * @param col    column index
 * @param image  pointer to image
 */
void m_image_write (matrix_t *M, int col, image_t *image)
{
	assert(M->rows == image->channels * image->height * image->width);

	int i;
	for ( i = 0; i < M->rows; i++ ) {
		image->pixels[i] = (unsigned char) elem(M, i, col);
	}
}

/**
 * Compute the covariance matrix of a matrix M, whose
 * columns are random variables and whose rows are
 * observations.
 *
 * If the columns of M are observations and the rows
 * of M are random variables, the covariance is:
 *
 *   C = 1/(N - 1) (M - mu * 1_N') (M - mu * 1_N')', N = M->cols
 *
 * If the columns of M are random variables and the
 * rows of M are observations, the covariance is:
 *
 *   C = 1/(N - 1) (M - 1_N * mu)' (M - 1_N * mu), N = M->rows
 *
 * @param M  pointer to matrix
 * @return pointer to covariance matrix of M
 */
matrix_t * m_covariance (matrix_t *M)
{
	// compute A = M - 1_N * mu
	matrix_t *A = m_copy(M);
	matrix_t *mu = m_mean_row(A);

	m_subtract_rows(A, mu);

	// compute C = 1/(N - 1) * A' * A
	matrix_t *C = m_zeros(A->cols, A->cols);

	// C := alpha * A' * A + beta * C, alpha = 1, beta = 0
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		A->cols, A->cols, A->rows,
		1, A->data, A->rows, A->data, A->rows,
		0, C->data, C->rows);

	precision_t c = (M->rows > 1)
		? M->rows - 1
		: 1;
	m_elem_mult(C, 1 / c);

	m_free(A);
	m_free(mu);

	return C;
}

/**
 * Compute the diagonal matrix of a vector.
 *
 * @param v  pointer to vector
 * @return pointer to diagonal matrix of v
 */
matrix_t * m_diagonalize (matrix_t *v)
{
	assert(v->rows == 1 || v->cols == 1);

	int n = (v->rows == 1)
		? v->cols
		: v->rows;
    matrix_t *D = m_zeros(n, n);

    int i;
    for ( i = 0; i < n; i++ ) {
        elem(D, i, i) = v->data[i];
    }

    return D;
}

/**
 * Compute the COS distance between two column vectors.
 *
 * COS is the cosine angle:
 * d_cos(x, y) = -x * y / (||x|| * ||y||)
 *
 * @param A  pointer to matrix
 * @param i  column index of A
 * @param B  pointer to matrix
 * @param j  column index of B
 * @return COS distance between A_i and B_j
 */
precision_t m_dist_COS (matrix_t *A, int i, matrix_t *B, int j)
{
	assert(A->rows == B->rows);

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

	return -x_dot_y / sqrt(abs_x * abs_y);
}

/**
 * Compute the L1 distance between two column vectors.
 *
 * L1 is the Euclidean distance:
 * d_L1(x, y) = ||x - y||
 *
 * @param A  pointer to matrix
 * @param i  column index of A
 * @param B  pointer to matrix
 * @param j  column index of B
 * @return L1 distance between A_i and B_j
 */
precision_t m_dist_L1 (matrix_t *A, int i, matrix_t *B, int j)
{
	return sqrt(m_dist_L2(A, i, B, j));
}

/**
 * Compute the L2 distance between two column vectors.
 *
 * L2 is the square of the Euclidean distance:
 * d_L2(x, y) = ||x - y||^2
 *
 * @param A  pointer to matrix
 * @param i  column index of A
 * @param B  pointer to matrix
 * @param j  column index of B
 * @return L2 distance between A_i and B_j
 */
precision_t m_dist_L2 (matrix_t *A, int i, matrix_t *B, int j)
{
	assert(A->rows == B->rows);

	precision_t dist = 0;

	int k;
	for ( k = 0; k < A->rows; k++ ) {
		precision_t diff = elem(A, k, i) - elem(B, k, j);
		dist += diff * diff;
	}

	return dist;
}

/**
 * Compute the real eigenvalues and right eigenvectors of a matrix.
 *
 * The eigenvalues are returned as a column vector, and the
 * eigenvectors are returned as column vectors. The i-th
 * eigenvalue corresponds to the i-th column vector.
 *
 * @param M	        pointer to matrix, m-by-n
 * @param p_M_eval  pointer to store eigenvalues matrix, m-by-1
 * @param p_M_evec  pointer to store eigenvectors matrix, m-by-n
 */
void m_eigen (matrix_t *M, matrix_t **p_M_eval, matrix_t **p_M_evec)
{
	matrix_t *M_eval = m_initialize(M->rows, 1);
	matrix_t *M_evec = m_initialize(M->rows, M->cols);

	matrix_t *M_work = m_copy(M);
	precision_t *wi = (precision_t *)malloc(M->rows * sizeof(precision_t));

	LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V',
		M->cols, M_work->data, M->rows,  // input matrix
		M_eval->data, wi,                // real, imag eigenvalues
		NULL, M->rows,                   // left eigenvectors
		M_evec->data, M->rows);          // right eigenvectors

	m_free(M_work);
	free(wi);

	*p_M_eval = M_eval;
	*p_M_evec = M_evec;
}

/**
 * Compute the generalized eigenvalues and right eigenvectors of two
 * square matrices.
 *
 * The eigenvalues are returned as a column vector, and the
 * eigenvectors are returned as column vectors. The i-th
 * eigenvalue corresponds to the i-th column vector.
 *
 * @param A	        pointer to matrix, n-by-n
 * @param B	        pointer to matrix, n-by-n
 * @param p_J_eval  pointer to store eigenvalues matrix, n-by-1
 * @param p_J_evec  pointer to store eigenvectors matrix, n-by-n
 */
void m_eigen2 (matrix_t *A, matrix_t *B, matrix_t **p_J_eval, matrix_t **p_J_evec)
{
	assert(A->rows == A->cols && B->rows == B->cols);
	assert(A->rows == B->rows);

	matrix_t *J_eval = m_initialize(A->rows, 1);
	matrix_t *J_evec = m_initialize(A->rows, A->cols);

	matrix_t *A_work = m_copy(A);
	matrix_t *B_work = m_copy(B);
	precision_t *alphai = (precision_t *)malloc(A->rows * sizeof(precision_t));
	precision_t *beta = (precision_t *)malloc(A->rows * sizeof(precision_t));

	LAPACKE_dggev(LAPACK_COL_MAJOR, 'N', 'V',
		A->cols, A_work->data, A->rows, B_work->data, B->rows,
		J_eval->data, alphai, beta,      // eigenvalues (lambda = (alphar + i * alphai) / beta)
		NULL, A->rows,                   // left eigenvectors
		J_evec->data, A->rows);          // right eigenvectors

	int i;
	for ( i = 0; i < A->rows; i++ ) {
		elem(J_eval, i, 0) /= beta[i];
	}

	m_free(A_work);
	m_free(B_work);
	free(alphai);
	free(beta);

	*p_J_eval = J_eval;
	*p_J_evec = J_evec;
}

/**
 * Compute the inverse of a square matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to new matrix equal to M^-1
 */
matrix_t * m_inverse (matrix_t *M)
{
	assert(M->rows == M->cols);

	matrix_t *M_inv = m_copy(M);
	int *ipiv = (int *)malloc(M->rows * sizeof(int));

	LAPACKE_dgetrf(LAPACK_COL_MAJOR,
		M->rows, M->cols, M_inv->data, M->rows,
		ipiv);

	LAPACKE_dgetri(LAPACK_COL_MAJOR,
		M->cols, M_inv->data, M->rows,
		ipiv);

	free(ipiv);

	return M_inv;
}

/**
 * Get the mean column of a matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to mean column vector
 */
matrix_t * m_mean_column (matrix_t *M)
{
	matrix_t *a = m_zeros(M->rows, 1);

	int i, j;
	for ( i = 0; i < M->cols; i++ ) {
		for ( j = 0; j < M->rows; j++ ) {
			elem(a, j, 0) += elem(M, j, i);
		}
	}

	for ( i = 0; i < M->rows; i++ ) {
		elem(a, i, 0) /= M->cols;
	}

	return a;
}

/**
 * Get the mean row of a matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to mean row vector
 */
matrix_t * m_mean_row (matrix_t *M)
{
	matrix_t *a = m_zeros(1, M->cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(a, 0, j) += elem(M, i, j);
		}
	}

	for ( i = 0; i < M->cols; i++ ) {
		elem(a, 0, i) /= M->rows;
	}

	return a;
}

/**
 * Compute the 2-norm of a vector.
 *
 * @param v  pointer to vector
 * @return 2-norm of v
 */
precision_t m_norm(matrix_t *v)
{
	assert(v->rows == 1 || v->cols == 1);

    precision_t sum = 0;
	int n = (v->rows == 1)
		? v->cols
		: v->rows;

    int i;
    for ( i = 0; i < n; i++ ) {
        sum += v->data[i] * v->data[i];
    }

    return sqrt(sum);
}

/**
 * Get the product of two matrices.
 *
 * @param A  pointer to left matrix
 * @param B  pointer to right matrix
 * @return pointer to new matrix equal to A * B
 */
matrix_t * m_product (matrix_t *A, matrix_t *B)
{
	assert(A->cols == B->rows);

	matrix_t *C = m_zeros(A->rows, B->cols);

	// C := alpha * op(A) * op(B) + beta * C, alpha = 1, beta = 0
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		A->rows, B->cols, A->cols,
		1, A->data, A->rows, B->data, B->rows,
		0, C->data, C->rows);

	return C;
}

/**
 * Compute the principal square root of a symmetric matrix. That
 * is, compute X such that X * X = M and X is the unique square root
 * for which every eigenvalue has non-negative real part.
 *
 * @param M  pointer to symmetric matrix
 * @return pointer to square root matrix
 */
matrix_t * m_sqrtm (matrix_t *M)
{
	assert(M->rows == M->cols);

	// compute eigenvalues, eigenvectors
	matrix_t *M_work = m_copy(M);
	matrix_t *M_eval = m_initialize(M->rows, 1);
	matrix_t *M_evec = m_initialize(M->rows, M->cols);

	int num_eval;
	int *ISUPPZ = (int *)malloc(2 * M->rows * sizeof(int));

	LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'A', 'L',
		M->cols, M_work->data, M->rows,
		0, 0, 0, 0, LAPACKE_dlamch('S'),
		&num_eval, M_eval->data, M_evec->data, M_evec->rows,
		ISUPPZ);

	m_free(M_work);
	free(ISUPPZ);

	assert(num_eval == M->rows);

	// compute B = M_evec * sqrt(D),
	//   D = eigenvalues of M in the diagonal
	matrix_t *B = m_copy(M_evec);

	int i, j;
	for ( j = 0; j < B->cols; j++ ) {
		precision_t lambda = sqrt(elem(M_eval, j, 0));

		for ( i = 0; i < B->rows; i++ ) {
			elem(B, i, j) *= lambda;
		}
	}

	m_free(M_eval);

	// compute X = B * M_evec'
	// X := alpha * B * M_evec' + beta * X, alpha = 1, beta = 0
	matrix_t *X = m_initialize(B->rows, M_evec->rows);

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		B->rows, M_evec->rows, B->cols,
		1, B->data, B->rows, M_evec->data, M_evec->rows,
		0, X->data, X->rows);

	m_free(B);
	m_free(M_evec);

	return X;
}

/**
 * Get the transpose of a matrix.
 *
 * @param M  pointer to matrix
 * @return pointer to new transposed matrix
 */
matrix_t * m_transpose (matrix_t *M)
{
	matrix_t *T = m_initialize(M->cols, M->rows);

	int i, j;
	for ( i = 0; i < T->rows; i++ ) {
		for ( j = 0; j < T->cols; j++ ) {
			elem(T, i, j) = elem(M, j, i);
		}
	}

	return T;
}

/**
 * Add a matrix to another matrix.
 *
 * @param A  pointer to matrix
 * @param B  pointer to matrix
 */
void m_add (matrix_t *A, matrix_t *B)
{
	assert(A->rows == B->rows && A->cols == B->cols);

	int i, j;
	for ( i = 0; i < A->rows; i++ ) {
		for ( j = 0; j < A->cols; j++ ) {
			elem(A, i, j) += elem(B, i, j);
		}
	}
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
    assert(A->rows == B->rows);

    memcpy(&elem(A, 0, i), B->data, B->rows * sizeof(precision_t));
}

/**
 * Subtract a matrix from another matrix.
 *
 * @param A  pointer to matrix
 * @param B  pointer to matrix
 */
void m_subtract (matrix_t *A, matrix_t *B)
{
	assert(A->rows == B->rows && A->cols == B->cols);

	int i, j;
	for ( i = 0; i < A->rows; i++ ) {
		for ( j = 0; j < A->cols; j++ ) {
			elem(A, i, j) -= elem(B, i, j);
		}
	}
}

/**
 * Apply a function to each element of a matrix.
 *
 * @param M  pointer to a matrix
 * @param f  pointer to element-wise function
 */
void m_elem_apply (matrix_t * M, elem_func_t f)
{
    int i, j;

    for ( i = 0; i < M->rows; i++ ) {
        for ( j = 0; j < M->cols; j++ ) {
            elem(M, i, j) = f(elem(M, i, j));
        }
    }
}

/**
 * Multiply a matrix by a scalar.
 *
 * @param M  pointer to matrix
 * @param c  scalar
 */
void m_elem_mult (matrix_t *M, precision_t c)
{
	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) *= c;
		}
	}
}

/**
 * Shuffle the columns of a matrix.
 *
 * @param M  pointer to matrix
 */
void m_shuffle_columns (matrix_t *M)
{
	precision_t *temp = (precision_t *)malloc(M->rows * sizeof(precision_t));

	int i, j;
	for ( i = 0; i < M->cols - 1; i++ ) {
		// generate j such that i <= j < M->cols
		j = rand() % (M->cols - i) + i;

		// swap columns i and j
		if ( i != j ) {
			memcpy(temp, &elem(M, 0, i), M->rows * sizeof(precision_t));
			memcpy(&elem(M, 0, i), &elem(M, 0, j), M->rows * sizeof(precision_t));
			memcpy(&elem(M, 0, j), temp, M->rows * sizeof(precision_t));
		}
	}

	free(temp);
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
	assert(M->rows == a->rows && a->cols == 1);

	int i, j;
	for ( i = 0; i < M->cols; i++ ) {
		for ( j = 0; j < M->rows; j++ ) {
			elem(M, j, i) -= elem(a, j, 0);
		}
	}
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
	assert(M->cols == a->cols && a->rows == 1);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) -= elem(a, 0, j);
		}
	}
}
