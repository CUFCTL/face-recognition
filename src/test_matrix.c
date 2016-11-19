/**
 * @file test_matrix.c
 *
 * Test suite for the matrix library.
 *
 * Tests are based on examples in the MATLAB documentation
 * where appropriate.
 */
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"

#define ANSI_RED    "\x1b[31m"
#define ANSI_BOLD   "\x1b[1m"
#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RESET  "\x1b[0m"

typedef void (*test_func_t)(void);

int VERBOSE;

/**
 * Construct a matrix with arbitrary data.
 *
 * @param rows
 * @param cols
 * @param data
 * @return pointer to matrix
 */
matrix_t * m_initialize_data (int rows, int cols, precision_t *data)
{
	matrix_t *M = m_initialize(rows, cols);

	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = data[i * cols + j];
		}
	}

	cublas_set_matrix(M);

	return M;
}

/**
 * Determine whether two floating point values are equal.
 *
 * @param a
 * @param b
 * @return 1 if a = b, 0 otherwise
 */
int is_equal (precision_t a, precision_t b)
{
	static precision_t EPSILON = 10e-4;

	return fabs(a - b) < EPSILON;
}

/**
 * Determine whether two matrices are equal.
 *
 * @param A
 * @param B
 * @return 1 if A = B, 0 otherwise
 */
int m_equal (matrix_t *A, matrix_t *B)
{
	if ( A->rows != B->rows || A->cols != B->cols ) {
		return 0;
	}

	int i, j;
	for ( i = 0; i < A->rows; i++ ) {
		for ( j = 0; j < A->cols; j++ ) {
			if ( !is_equal(elem(A, i, j), elem(B, i, j)) ) {
				return 0;
			}
		}
	}

	return 1;
}

/**
 * Assert that two floating-point values are equal.
 *
 * @param a
 * @param b
 * @param name
 */
void assert_equal (precision_t a, precision_t b, const char *name)
{
	if ( is_equal(a, b) ) {
		printf(ANSI_GREEN "PASSED: %s" ANSI_RESET "\n", name);
	}
	else {
		printf(ANSI_RED ANSI_BOLD "FAILED: %s" ANSI_RESET "\n", name);
	}

	fflush(stdout);
}

/**
 * Assert that two matrices are equal.
 *
 * @param A
 * @param B
 * @param name
 */
void assert_equal_matrix (matrix_t *A, matrix_t *B, const char *name)
{
	if ( m_equal(A, B) ) {
		printf(ANSI_GREEN "PASSED: %s" ANSI_RESET "\n", name);
	}
	else {
		printf(ANSI_RED ANSI_BOLD "FAILED: %s" ANSI_RESET "\n", name);
	}

	fflush(stdout);
}

/**
 * Assert that a matrix M is equal to a test value.
 *
 * @param M
 * @param data
 * @param name
 */
void assert_matrix_value (matrix_t *M, precision_t *data, const char *name)
{
	matrix_t *M_test = m_initialize_data(M->rows, M->cols, data);

	assert_equal_matrix(M, M_test, name);

	m_free(M_test);
}

/**
 * Test identity matrix.
 */
void test_m_identity()
{
	precision_t I_data[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	matrix_t *I = m_identity(4);

	if ( VERBOSE ) {
		printf("I = eye(%d) = \n", I->rows);
		m_fprint(stdout, I);
	}

	assert_matrix_value(I, I_data, "eye(N)");

	m_free(I);
}

/**
 * Test ones matrix.
 */
void test_m_ones()
{
	precision_t X_data[] = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1
	};
	matrix_t *X = m_ones(4, 4);

	if ( VERBOSE ) {
		printf("X = ones(%d, %d) = \n", X->rows, X->cols);
		m_fprint(stdout, X);
	}

	assert_matrix_value(X, X_data, "ones(M, N)");

	m_free(X);
}

/**
 * Test random matrix.
 */
void test_m_random()
{
	precision_t X_data[] = {
		-2.4191,  0.7112, -0.3091,  0.5153,  1.0677,
		 0.0119,  1.8262, -0.1099, -0.5056,  0.5643,
		-1.2957, -0.6433,  1.8492,  0.7887,  0.1808,
		 0.2610,  0.2159,  1.0133,  0.4165,  0.5066,
		 0.2370,  0.9451, -2.8396, -1.4088, -0.2007
	};
	matrix_t *X = m_random(5, 5);

	if ( VERBOSE ) {
		printf("X = randn(%d, %d) = \n", X->rows, X->cols);
		m_fprint(stdout, X);
	}

	assert_matrix_value(X, X_data, "randn(M, N)");

	m_free(X);
}

/**
 * Test zero matrix.
 */
void test_m_zeros()
{
	precision_t X_data[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	matrix_t *X = m_zeros(4, 4);

	if ( VERBOSE ) {
		printf("X = zeros(%d, %d) = \n", X->rows, X->cols);
		m_fprint(stdout, X);
	}

	assert_matrix_value(X, X_data, "zeros(M, N)");

	m_free(X);
}

/**
 * Test matrix copy.
 */
void test_m_copy()
{
	precision_t A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	matrix_t *A = m_initialize_data(4, 4, A_data);
	matrix_t *C = m_copy(A);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("C = A = \n");
		m_fprint(stdout, C);
	}

	assert_equal_matrix(A, C, "A(:, :)");

	m_free(A);
	m_free(C);
}

/**
 * Test matrix column copy.
 */
void test_m_copy_columns()
{
	precision_t A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	precision_t C_data[] = {
		 2,  3,
		11, 10,
		 7,  6,
		14, 15
	};
	matrix_t *A = m_initialize_data(4, 4, A_data);

	int i = 1;
	int j = 3;
	matrix_t *C = m_copy_columns(A, i, j);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("C = A(:, %d:%d) = \n", i + 1, j);
		m_fprint(stdout, C);
	}

	assert_matrix_value(C, C_data, "A(:, 1:3)");

	m_free(A);
	m_free(C);
}

/**
 * Test matrix row copy.
 */
void test_m_copy_rows()
{
	precision_t A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	precision_t C_data[] = {
		 5, 11, 10,  8,
		 9,  7,  6, 12
	};
	matrix_t *A = m_initialize_data(4, 4, A_data);

	int i = 1;
	int j = 3;
	matrix_t *C = m_copy_rows(A, i, j);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("C = A(%d:%d, :) = \n", i + 1, j);
		m_fprint(stdout, C);
	}

	assert_matrix_value(C, C_data, "A(1:3, :)");

	m_free(A);
	m_free(C);
}

/**
 * Test the matrix convariance.
 */
void test_m_covariance()
{
	precision_t A_data[] = {
		5,  0,  3,  7,
		1, -5,  7,  3,
		4,  9,  8, 10
	};
	precision_t C_data[] = {
		 4.3333,  8.8333, -3.0000,  5.6667,
		 8.8333, 50.3333,  6.5000, 24.1667,
		-3.0000,  6.5000,  7.0000,  1.0000,
		 5.6667, 24.1667,  1.0000, 12.3333
	};
	matrix_t *A = m_initialize_data(3, 4, A_data);
	matrix_t *C = m_covariance(A);

	cublas_get_matrix(C);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("cov(A) = \n");
		m_fprint(stdout, C);
	}

	assert_matrix_value(C, C_data, "cov(A)");

	m_free(A);
	m_free(C);
}

/**
 * Test the diagonal matrix.
 */
void test_m_diagonalize()
{
	precision_t v_data[] = {
		2, 1, -1, -2, -5
	};
	precision_t D_data[] = {
		2,  0,  0,  0,  0,
		0,  1,  0,  0,  0,
		0,  0, -1,  0,  0,
		0,  0,  0, -2,  0,
		0,  0,  0,  0, -5
	};
	matrix_t *v = m_initialize_data(1, 5, v_data);
	matrix_t *D = m_diagonalize(v);

	if ( VERBOSE ) {
		printf("v = \n");
		m_fprint(stdout, v);

		printf("diag(v) = \n");
		m_fprint(stdout, D);
	}

	assert_matrix_value(D, D_data, "diag(v)");

	m_free(v);
	m_free(D);
}

/**
 * Test the vector distance functions.
 */
void test_m_distance()
{
	precision_t a_data[] = {
		1,
		0,
		0
	};
	precision_t b_data[] = {
		0,
		1,
		0
	};
	matrix_t *a = m_initialize_data(3, 1, a_data);
	matrix_t *b = m_initialize_data(3, 1, b_data);
	precision_t dist_COS = m_dist_COS(a, 0, b, 0);
	precision_t dist_L1 = m_dist_L1(a, 0, b, 0);
	precision_t dist_L2 = m_dist_L2(a, 0, b, 0);

	if ( VERBOSE ) {
		printf("a = \n");
		m_fprint(stdout, a);

		printf("b = \n");
		m_fprint(stdout, b);

		printf("d_COS(a, b) = % 8.4lf\n", dist_COS);
		printf("d_L1 (a, b) = % 8.4lf\n", dist_L1);
		printf("d_L2 (a, b) = % 8.4lf\n", dist_L2);
	}

	assert_equal(dist_COS, 0.0000, "d_COS(a, b)");
	assert_equal(dist_L1, 1.4142, "d_L1(a, b)");
	assert_equal(dist_L2, 2.0000, "d_L2(a, b)");

	m_free(a);
	m_free(b);
}

/**
 * Test eigenvalues, eigenvectors.
 */
void test_m_eigen()
{
	precision_t M_data[] = {
		1.0000, 0.5000, 0.3333, 0.2500,
		0.5000, 1.0000, 0.6667, 0.5000,
		0.3333, 0.6667, 1.0000, 0.7500,
		0.2500, 0.5000, 0.7500, 1.0000
	};
	precision_t M_eval_data[] = {
		0.2078,
		0.4078,
		0.8482,
		2.5362
	};
	precision_t M_evec_data[] = {
		 0.0694, -0.4422, -0.8105,  0.3778,
		-0.3619,  0.7420, -0.1877,  0.5322,
		 0.7694,  0.0487,  0.3010,  0.5614,
		-0.5218, -0.5015,  0.4661,  0.5088
	};
	matrix_t *M = m_initialize_data(4, 4, M_data);
	matrix_t *M_eval;
	matrix_t *M_evec;

	m_eigen(M, &M_eval, &M_evec);

	if ( VERBOSE ) {
		printf("M = \n");
		m_fprint(stdout, M);

		printf("eigenvalues of M = \n");
		m_fprint(stdout, M_eval);

		printf("eigenvectors of M = \n");
		m_fprint(stdout, M_evec);
	}

	assert_matrix_value(M_eval, M_eval_data, "eigenvalues of M");
	assert_matrix_value(M_evec, M_evec_data, "eigenvectors of M");

	m_free(M);
	m_free(M_eval);
	m_free(M_evec);
}

/**
 * Test generalized eigenvalues, eigenvectors for two matrices.
 */
void test_m_eigen2()
{
	precision_t A_data[] = {
		1, 1, 1,
		1, 1, 1,
		1, 1, 1
	};
	precision_t B_data[] = {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1
	};
	precision_t J_eval_data[] = {
		0.0000,
		0.0000,
		3.0000
	};
	precision_t J_evec_data[] = {
		 0.4082,  0.7071,  0.5774,
		 0.4082, -0.7071,  0.5774,
		-0.8165,  0.0000,  0.5774
	};
	matrix_t *A = m_initialize_data(3, 3, A_data);
	matrix_t *B = m_initialize_data(3, 3, B_data);
	matrix_t *J_eval;
	matrix_t *J_evec;

	m_eigen2(A, B, &J_eval, &J_evec);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = \n");
		m_fprint(stdout, B);

		printf("eigenvalues of A, B = \n");
		m_fprint(stdout, J_eval);

		printf("eigenvectors of A, B = \n");
		m_fprint(stdout, J_evec);
	}

	assert_matrix_value(J_eval, J_eval_data, "eigenvalues of A, B");
	assert_matrix_value(J_evec, J_evec_data, "eigenvectors of A, B");

	m_free(A);
	m_free(B);
	m_free(J_eval);
	m_free(J_evec);
}

/**
 * Test matrix inverse.
 */
void test_m_inverse()
{
	precision_t X_data[] = {
		 1,  0,  2,
		-1,  5,  0,
		 0,  3, -9
	};
	precision_t Y_data[] = {
		0.8824, -0.1176,  0.1961,
		0.1765,  0.1765,  0.0392,
		0.0588,  0.0588, -0.0980
	};
	matrix_t *X = m_initialize_data(3, 3, X_data);
	matrix_t *Y = m_inverse(X);

	if ( VERBOSE ) {
		printf("X = \n");
		m_fprint(stdout, X);

		printf("Y = inv(X) = \n");
		m_fprint(stdout, Y);
	}

	assert_matrix_value(Y, Y_data, "inv(X)");

	m_free(X);
	m_free(Y);
}

/**
 * Test matrix mean column.
 */
void test_m_mean_column()
{
	precision_t A_data[] = {
		0, 1, 1,
		2, 3, 2
	};
	precision_t m_data[] = {
		0.6667,
		2.3333
	};
	matrix_t *A = m_initialize_data(2, 3, A_data);
	matrix_t *m = m_mean_column(A);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("mean(A, 2) = \n");
		m_fprint(stdout, m);
	}

	assert_matrix_value(m, m_data, "mean(A, 2)");

	m_free(A);
	m_free(m);
}

/**
 * Test matrix mean row.
 */
void test_m_mean_row()
{
	precision_t A_data[] = {
		0, 1, 1,
		2, 3, 2,
		1, 3, 2,
		4, 2, 2
	};
	precision_t m_data[] = {
		1.7500, 2.2500, 1.7500
	};
	matrix_t *A = m_initialize_data(4, 3, A_data);
	matrix_t *m = m_mean_row(A);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("mean(A, 1) = \n");
		m_fprint(stdout, m);
	}

	assert_matrix_value(m, m_data, "mean(A, 1)");

	m_free(A);
	m_free(m);
}

/**
 * Test vector norm.
 */
void test_m_norm()
{
	precision_t v_data[] = {
		-2, 3, 1
	};
	matrix_t *v = m_initialize_data(1, 3, v_data);
	precision_t n = m_norm(v);

	if ( VERBOSE ) {
		printf("v = \n");
		m_fprint(stdout, v);

		printf("norm(v) = % 8.4lf\n", n);
	}

	assert_equal(n, 3.7417, "norm(v)");

	m_free(v);
}

/**
 * Test matrix product.
 */
void test_m_product()
{
	// multiply two vectors, A * B
	// multiply two vectors, B * A
	precision_t A1_data[] = {
		1, 1, 0, 0
	};
	precision_t B1_data[] = {
		1,
		2,
		3,
		4
	};
	precision_t C1_data[] = {
		3
	};
	precision_t C2_data[] = {
		1, 1, 0, 0,
		2, 2, 0, 0,
		3, 3, 0, 0,
		4, 4, 0, 0
	};
	matrix_t *A1 = m_initialize_data(1, 4, A1_data);
	matrix_t *B1 = m_initialize_data(4, 1, B1_data);
	matrix_t *C1 = m_product(A1, B1);
	matrix_t *C2 = m_product(B1, A1);

	cublas_get_matrix(C1);
	cublas_get_matrix(C2);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A1);

		printf("B = \n");
		m_fprint(stdout, B1);

		printf("A * B = \n");
		m_fprint(stdout, C1);

		printf("B * A = \n");
		m_fprint(stdout, C2);
	}

	assert_matrix_value(C1, C1_data, "A1 * B1");
	assert_matrix_value(C2, C2_data, "B1 * A1");

	m_free(A1);
	m_free(B1);
	m_free(C1);
	m_free(C2);

	// multiply two matrices
	precision_t A2_data[] = {
		1, 3, 5,
		2, 4, 7
	};
	precision_t B2_data[] = {
		-5, 8, 11,
		 3, 9, 21,
		 4, 0,  8
	};
	precision_t C3_data[] = {
		24, 35, 114,
		30, 52, 162
	};
	matrix_t *A2 = m_initialize_data(2, 3, A2_data);
	matrix_t *B2 = m_initialize_data(3, 3, B2_data);
	matrix_t *C3 = m_product(A2, B2);

	cublas_get_matrix(C3);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A2);

		printf("B = \n");
		m_fprint(stdout, B2);

		printf("A * B = \n");
		m_fprint(stdout, C3);
	}

	assert_matrix_value(C3, C3_data, "A2 * B2");

	m_free(A2);
	m_free(B2);
	m_free(C3);
}

/**
 * Test matrix square root.
 */
void test_m_sqrtm()
{
	precision_t A_data[] = {
		 5, -4,  1,  0,  0,
		-4,  6, -4,  1,  0,
		 1, -4,  6, -4,  1,
		 0,  1, -4,  6, -4,
		 0,  0,  1, -4,  6
	};
	precision_t X_data[] = {
		 2.0015, -0.9971,  0.0042,  0.0046,  0.0032,
		-0.9971,  2.0062, -0.9904,  0.0118,  0.0094,
		 0.0042, -0.9904,  2.0171, -0.9746,  0.0263,
		 0.0046,  0.0118, -0.9746,  2.0503, -0.9200,
		 0.0032,  0.0094,  0.0263, -0.9200,  2.2700
	};
	matrix_t *A = m_initialize_data(5, 5, A_data);
	matrix_t *X = m_sqrtm(A);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("X = sqrtm(A) = \n");
		m_fprint(stdout, X);
	}

	assert_matrix_value(X, X_data, "sqrtm(A)");

	m_free(A);
	m_free(X);
}

/**
 * Test matrix transpose.
 */
void test_m_transpose()
{
	precision_t A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	precision_t B_data[] = {
		16,  5,  9,  4,
		 2, 11,  7, 14,
		 3, 10,  6, 15,
		13,  8, 12,  1
	};
	matrix_t *A = m_initialize_data(4, 4, A_data);
	matrix_t *B = m_transpose(A);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = A' = \n");
		m_fprint(stdout, B);
	}

	assert_matrix_value(B, B_data, "A'");

	m_free(A);
	m_free(B);
}

/**
 * Test matrix addition.
 */
void test_m_add()
{
	precision_t A_data1[] = {
		1, 0,
		2, 4
	};
	precision_t A_data2[] = {
		6, 9,
		4, 5
	};
	precision_t B_data[] = {
		5, 9,
		2, 1
	};
	matrix_t *A = m_initialize_data(2, 2, A_data1);
	matrix_t *B = m_initialize_data(2, 2, B_data);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = \n");
		m_fprint(stdout, B);
	}

	m_add(A, B);

	if ( VERBOSE ) {
		printf("A + B = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "A + B");

	m_free(A);
	m_free(B);
}

/**
 * Test matrix column assingment.
 */
void test_m_assign_column()
{
	precision_t A_data1[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	precision_t A_data2[] = {
		16,  2,  0, 13,
		 5, 11,  0,  8,
		 9,  7,  0, 12,
		 4, 14,  0,  1
	};
	precision_t B_data[] = {
		0,
		0,
		0,
		0
	};
	matrix_t *A = m_initialize_data(4, 4, A_data1);
	matrix_t *B = m_initialize_data(4, 1, B_data);
	int i = 2;
	int j = 0;

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = \n");
		m_fprint(stdout, B);

		printf("A(:, %d) = B(:, %d)\n", i, j);
	}

	m_assign_column(A, i, B, j);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "A(:, 2) = B(:, 0)");

	m_free(A);
	m_free(B);
}

/**
 * Test matrix row assingment.
 */
void test_m_assign_row()
{
	precision_t A_data1[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	precision_t A_data2[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 0,  0,  0,  0,
		 4, 14, 15,  1
	};
	precision_t B_data[] = {
		0, 0, 0, 0
	};
	matrix_t *A = m_initialize_data(4, 4, A_data1);
	matrix_t *B = m_initialize_data(1, 4, B_data);
	int i = 2;
	int j = 0;

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = \n");
		m_fprint(stdout, B);

		printf("A(%d, :) = B(%d, :)\n", i, j);
	}

	m_assign_row(A, i, B, j);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "A(2, :) = B(0, :)");

	m_free(A);
	m_free(B);
}

/**
 * Test matrix subtraction.
 */
void test_m_subtract()
{
	precision_t A_data1[] = {
		1, 0,
		2, 4
	};
	precision_t A_data2[] = {
		-4, -9,
		 0,  3
	};
	precision_t B_data[] = {
		5, 9,
		2, 1
	};
	matrix_t *A = m_initialize_data(2, 2, A_data1);
	matrix_t *B = m_initialize_data(2, 2, B_data);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);

		printf("B = \n");
		m_fprint(stdout, B);
	}

	m_subtract(A, B);

	if ( VERBOSE ) {
		printf("A - B = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "A - B");

	m_free(A);
	m_free(B);
}

/**
 * Test matrix element-wise function application.
 */
void test_m_elem_apply()
{
	precision_t A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	precision_t A_data2[] = {
		1.0000, 0.0000, 1.4142,
		1.7321, 1.0000, 2.0000
	};
	matrix_t *A = m_initialize_data(2, 3, A_data1);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);
	}

	m_elem_apply(A, sqrt);

	if ( VERBOSE ) {
		printf("sqrt(A) = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "sqrt(A)");

	m_free(A);
}

/**
 * Test matrix multiplication by scalar.
 */
void test_m_elem_mult()
{
	precision_t A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	precision_t A_data2[] = {
		3, 0, 6,
		9, 3, 12
	};
	matrix_t *A = m_initialize_data(2, 3, A_data1);
	precision_t c = 3;

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);
	}

	m_elem_mult(A, c);

	if ( VERBOSE ) {
		printf("%lg * A = \n", c);
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "3 * A");

	m_free(A);
}

/**
 * Test matrix column shuffling.
 */
void test_m_shuffle_columns()
{
	precision_t A_data1[] = {
		1, 2, 3, 4,
		5, 6, 7, 8
	};
	precision_t A_data2[] = {
		4, 3, 1, 2,
		8, 7, 5, 6
	};
	matrix_t *A = m_initialize_data(2, 4, A_data1);

	if ( VERBOSE ) {
		printf("A = \n");
		m_fprint(stdout, A);
	}

	m_shuffle_columns(A);

	if ( VERBOSE ) {
		printf("A(:, randperm(size(A, 2))) = \n");
		m_fprint(stdout, A);
	}

	assert_matrix_value(A, A_data2, "A(:, randperm(size(A, 2)))");

	m_free(A);
}

/**
 * Test matrix column subtraction.
 */
void test_m_subtract_columns()
{
	precision_t M_data1[] = {
		0, 2, 1, 4,
		1, 3, 3, 2,
		1, 2, 2, 2
	};
	precision_t M_data2[] = {
		0, 2, 1, 4,
		0, 2, 2, 1,
		0, 1, 1, 1
	};
	precision_t a_data[] = {
		0,
		1,
		1
	};
	matrix_t *M = m_initialize_data(3, 4, M_data1);
	matrix_t *a = m_initialize_data(3, 1, a_data);;

	if ( VERBOSE ) {
		printf("M = \n");
		m_fprint(stdout, M);

		printf("a = \n");
		m_fprint(stdout, a);
	}

	m_subtract_columns(M, a);

	if ( VERBOSE ) {
		printf("M - a * 1_N' = \n");
		m_fprint(stdout, M);
	}

	assert_matrix_value(M, M_data2, "M - a * 1_N'");

	m_free(M);
	m_free(a);
}

/**
 * Test matrix row subtraction.
 */
void test_m_subtract_rows()
{
	precision_t M_data1[] = {
		0, 2, 1, 4,
		1, 3, 3, 2,
		1, 2, 2, 2
	};
	precision_t M_data2[] = {
		0,  0,  0,  0,
		1,  1,  2, -2,
		1,  0,  1, -2
	};
	precision_t a_data[] = {
		0, 2, 1, 4
	};
	matrix_t *M = m_initialize_data(3, 4, M_data1);
	matrix_t *a = m_initialize_data(1, 4, a_data);

	if ( VERBOSE ) {
		printf("M = \n");
		m_fprint(stdout, M);

		printf("a = \n");
		m_fprint(stdout, a);
	}

	m_subtract_rows(M, a);

	if ( VERBOSE ) {
		printf("M - 1_N * a = \n");
		m_fprint(stdout, M);
	}

	assert_matrix_value(M, M_data2, "M - 1_N * a");

	m_free(M);
	m_free(a);
}

int main (int argc, char **argv)
{
	// parse command-line arguments
	struct option long_options[] = {
		{ "verbose", no_argument, 0, 'v' },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case 'v':
			VERBOSE = 1;
			break;
		}
	}

	// run tests
	test_func_t tests[] = {
		test_m_identity,
		test_m_ones,
		test_m_random,
		test_m_zeros,
		test_m_copy,
		test_m_copy_columns,
		test_m_copy_rows,
		test_m_covariance,
		test_m_diagonalize,
		test_m_distance,
		test_m_eigen,
		test_m_eigen2,
		test_m_inverse,
		test_m_mean_column,
		test_m_mean_row,
		test_m_norm,
		test_m_product,
		test_m_sqrtm,
		test_m_transpose,
		test_m_add,
		test_m_subtract,
		test_m_assign_column,
		test_m_assign_row,
		test_m_elem_apply,
		test_m_elem_mult,
		test_m_shuffle_columns,
		test_m_subtract_columns,
		test_m_subtract_rows
	};
	int num_tests = sizeof(tests) / sizeof(test_func_t);

	int i;
	for ( i = 0; i < num_tests; i++ ) {
		test_func_t test = tests[i];

		printf("TEST %d\n\n", i + 1);

		test();
		putchar('\n');
	}

	return 0;
}
