/**
 * @file test_matrix.cpp
 *
 * Test suite for the matrix library.
 *
 * Tests are based on examples in the MATLAB documentation
 * where appropriate.
 */
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "logger.h"
#include "matrix.h"

#ifdef __NVCC__
	#include "magma_v2.h"
#endif

#define ANSI_RED    "\x1b[31m"
#define ANSI_BOLD   "\x1b[1m"
#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RESET  "\x1b[0m"

typedef void (*test_func_t)(void);

/**
 * Determine whether two floating point values are equal.
 *
 * @param a
 * @param b
 * @return 1 if a = b, 0 otherwise
 */
int is_equal(precision_t a, precision_t b)
{
	static precision_t EPSILON = 1e-4;

	return fabsf(a - b) < EPSILON;
}

/**
 * Determine whether two matrices are equal.
 *
 * @param A
 * @param B
 * @return 1 if A = B, 0 otherwise
 */
int m_equal(const Matrix& A, const Matrix& B)
{
	if ( A.rows != B.rows || A.cols != B.cols ) {
		return 0;
	}

	int i, j;
	for ( i = 0; i < A.rows; i++ ) {
		for ( j = 0; j < A.cols; j++ ) {
			if ( !is_equal(ELEM(A, i, j), ELEM(B, i, j)) ) {
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
void assert_equal(precision_t a, precision_t b, const char *name)
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
void assert_equal_matrix(const Matrix& A, const Matrix& B, const char *name)
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
void assert_matrix_value(const Matrix& M, precision_t *data, const char *name)
{
	Matrix M_test("M_test", M.rows, M.cols, data);

	assert_equal_matrix(M, M_test, name);
}

/**
 * Test identity matrix.
 */
void test_identity()
{
	precision_t I_data[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	Matrix I = Matrix::identity("I", 4);

	if ( LOGGER(LL_VERBOSE) ) {
		I.print(stdout);
	}

	assert_matrix_value(I, I_data, "eye(N)");
}

/**
 * Test ones matrix.
 */
void test_ones()
{
	precision_t X_data[] = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1
	};
	Matrix X = Matrix::ones("X", 4, 4);

	if ( LOGGER(LL_VERBOSE) ) {
		X.print(stdout);
	}

	assert_matrix_value(X, X_data, "ones(M, N)");
}

/**
 * Test random matrix.
 */
void test_random()
{
	precision_t X_data[] = {
		-2.4191,  0.7112, -0.3091,  0.5153,  1.0677,
		 0.0119,  1.8262, -0.1099, -0.5056,  0.5643,
		-1.2957, -0.6433,  1.8492,  0.7887,  0.1808,
		 0.2610,  0.2159,  1.0133,  0.4165,  0.5066,
		 0.2370,  0.9451, -2.8396, -1.4088, -0.2007
	};
	Matrix X = Matrix::random("X", 5, 5);

	if ( LOGGER(LL_VERBOSE) ) {
		X.print(stdout);
	}

	assert_matrix_value(X, X_data, "randn(M, N)");
}

/**
 * Test zero matrix.
 */
void test_zeros()
{
	precision_t X_data[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	Matrix X = Matrix::zeros("X", 4, 4);

	if ( LOGGER(LL_VERBOSE) ) {
		X.print(stdout);
	}

	assert_matrix_value(X, X_data, "zeros(M, N)");
}

/**
 * Test matrix copy.
 */
void test_copy()
{
	precision_t A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	Matrix A("A", 4, 4, A_data);
	Matrix C("C", A);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		C.print(stdout);
	}

	assert_equal_matrix(A, C, "A(:, :)");
}

/**
 * Test matrix column copy.
 */
void test_copy_columns()
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
	Matrix A("A", 4, 4, A_data);

	int i = 1;
	int j = 3;
	Matrix C("C", A, i, j);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		C.print(stdout);
	}

	assert_matrix_value(C, C_data, "A(:, i:j)");
}

/**
 * Test the argmax function.
 */
void test_argmax()
{
	precision_t v_data[] = {
		23, 42, 37, 18, 52
	};
	Matrix v("v", 1, 5, v_data);

	int index = v.argmax();

	if ( LOGGER(LL_VERBOSE) ) {
		v.print(stdout);

		printf("argmax(%s) = %d\n", v.name, index);
	}

	assert_equal(index, 4, "argmax(v)");
}

/**
 * Test the diagonal matrix.
 */
void test_diagonalize()
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
	Matrix v("v", 1, 5, v_data);
	Matrix D = v.diagonalize("D");

	if ( LOGGER(LL_VERBOSE) ) {
		v.print(stdout);

		D.print(stdout);
	}

	assert_matrix_value(D, D_data, "diag(v)");
}

/**
 * Test eigenvalues, eigenvectors.
 */
void test_eigen()
{
	precision_t M_data[] = {
		1.0000, 0.5000, 0.3333, 0.2500,
		0.5000, 1.0000, 0.6667, 0.5000,
		0.3333, 0.6667, 1.0000, 0.7500,
		0.2500, 0.5000, 0.7500, 1.0000
	};
	precision_t V_data[] = {
		 0.0694, -0.4422, -0.8105,  0.3778,
		-0.3619,  0.7420, -0.1877,  0.5322,
		 0.7694,  0.0487,  0.3010,  0.5614,
		-0.5218, -0.5015,  0.4661,  0.5088
	};
	precision_t D_data[] = {
		0.2078, 0.0000, 0.0000, 0.0000,
		0.0000, 0.4078, 0.0000, 0.0000,
		0.0000, 0.0000, 0.8482, 0.0000,
		0.0000, 0.0000, 0.0000, 2.5362
	};
	Matrix M("M", 4, 4, M_data);
	Matrix V;
	Matrix D;

	M.eigen("V", "D", M.rows, V, D);

	if ( LOGGER(LL_VERBOSE) ) {
		M.print(stdout);
		V.print(stdout);
		D.print(stdout);
	}

	assert_matrix_value(V, V_data, "eigenvectors of M");
	assert_matrix_value(D, D_data, "eigenvalues of M");
}

/**
 * Test generalized eigenvalues, eigenvectors for two matrices.
 */
void test_eigen2()
{
	precision_t A_data[] = {
		 0.5377,  0.8622, -0.4336,
		 0.8622,  0.3188,  0.3426,
		-0.4336,  0.3426,  3.5784
	};
	precision_t B_data[] = {
		 2, -1,  0,
		-1,  2, -1,
		 0, -1,  2
	};
	precision_t V_data[] = {
		 0.3714,  0.6224,  0.4740,
		-0.4204,  0.4574,  0.7836,
		 0.0606, -0.2616,  0.8233
	};
	precision_t D_data[] = {
		-0.1626,  0.0000,  0.0000,
		 0.0000,  1.0700,  0.0000,
		 0.0000,  0.0000,  3.4864
	};
	Matrix A("A", 3, 3, A_data);
	Matrix B("B", 3, 3, B_data);
	Matrix V;
	Matrix D;

	Matrix B_inv = B.inverse("inv(B)");
	Matrix J = B_inv * A;

	J.eigen("V", "D", J.rows, V, D);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
		V.print(stdout);
		D.print(stdout);
	}

	assert_matrix_value(V, V_data, "eigenvectors of A, B");
	assert_matrix_value(D, D_data, "eigenvalues of A, B");
}

/**
 * Test matrix inverse.
 */
void test_inverse()
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
	Matrix X("X", 3, 3, X_data);
	Matrix Y = X.inverse("Y");

	if ( LOGGER(LL_VERBOSE) ) {
		X.print(stdout);
		Y.print(stdout);
	}

	assert_matrix_value(Y, Y_data, "inv(X)");
}

/**
 * Test matrix mean column.
 */
void test_mean_column()
{
	precision_t A_data[] = {
		0, 1, 1,
		2, 3, 2
	};
	precision_t m_data[] = {
		0.6667,
		2.3333
	};
	Matrix A("A", 2, 3, A_data);
	Matrix m = A.mean_column("m");

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		m.print(stdout);
	}

	assert_matrix_value(m, m_data, "mean(A, 2)");
}

/**
 * Test matrix mean row.
 */
void test_mean_row()
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
	Matrix A("A", 4, 3, A_data);
	Matrix m = A.mean_row("m");

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		m.print(stdout);
	}

	assert_matrix_value(m, m_data, "mean(A, 1)");
}

/**
 * Test vector norm.
 */
void test_norm()
{
	precision_t v_data[] = {
		-2, 3, 1
	};
	Matrix v("v", 1, 3, v_data);
	precision_t n = v.norm();

	if ( LOGGER(LL_VERBOSE) ) {
		v.print(stdout);

		printf("norm(%s) = %g\n", v.name, n);
	}

	assert_equal(n, 3.7417, "norm(v)");
}

/**
 * Test matrix product.
 */
void test_product()
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
	Matrix A1("A1", 1, 4, A1_data);
	Matrix B1("B1", 4, 1, B1_data);
	Matrix C1 = A1 * B1;
	Matrix C2 = B1 * A1;

	if ( LOGGER(LL_VERBOSE) ) {
		A1.print(stdout);
		B1.print(stdout);
		C1.print(stdout);
		C2.print(stdout);
	}

	assert_matrix_value(C1, C1_data, "A1 * B1");
	assert_matrix_value(C2, C2_data, "B1 * A1");

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
	Matrix A2("A2", 2, 3, A2_data);
	Matrix B2("B2", 3, 3, B2_data);
	Matrix C3 = A2 * B2;

	if ( LOGGER(LL_VERBOSE) ) {
		A2.print(stdout);
		B2.print(stdout);
		C3.print(stdout);
	}

	assert_matrix_value(C3, C3_data, "A2 * B2");
}

/**
 * Test vector sum.
 */
void test_sum()
{
	precision_t v_data[] = {
		-2, 3, 1
	};
	Matrix v("v", 1, 3, v_data);
	precision_t s = v.sum();

	if ( LOGGER(LL_VERBOSE) ) {
		v.print(stdout);

		printf("sum(%s) = %g\n", v.name, s);
	}

	assert_equal(s, 2, "sum(v)");
}

/**
 * Test matrix transpose.
 */
void test_transpose()
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
	Matrix A("A", 4, 4, A_data);
	Matrix B = A.transpose("B");

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
	}

	assert_matrix_value(B, B_data, "A'");
}

/**
 * Test matrix addition.
 */
void test_add()
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
	Matrix A("A", 2, 2, A_data1);
	Matrix B("B", 2, 2, B_data);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
	}

	A += B;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "A + B");
}

/**
 * Test matrix column assingment.
 */
void test_assign_column()
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
	Matrix A("A", 4, 4, A_data1);
	Matrix B("B", 4, 1, B_data);
	int i = 2;
	int j = 0;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
	}

	A.assign_column(i, B, j);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "A(:, i) = B(:, j)");
}

/**
 * Test matrix row assingment.
 */
void test_assign_row()
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
	Matrix A("A", 4, 4, A_data1);
	Matrix B("B", 1, 4, B_data);
	int i = 2;
	int j = 0;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
	}

	A.assign_row(i, B, j);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "A(i, :) = B(j, :)");
}

/**
 * Test matrix element-wise function application.
 */
void test_elem_apply()
{
	precision_t A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	precision_t A_data2[] = {
		1.0000, 0.0000, 1.4142,
		1.7321, 1.0000, 2.0000
	};
	Matrix A("A", 2, 3, A_data1);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	A.elem_apply(sqrtf);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "sqrt(A)");
}

/**
 * Test matrix multiplication by scalar.
 */
void test_elem_mult()
{
	precision_t A_data1[] = {
		1, 0, 2,
		3, 1, 4
	};
	precision_t A_data2[] = {
		3, 0, 6,
		9, 3, 12
	};
	Matrix A("A", 2, 3, A_data1);
	precision_t c = 3;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	A *= c;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "c * A");
}

/**
 * Test matrix subtraction.
 */
void test_subtract()
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
	Matrix A("A", 2, 2, A_data1);
	Matrix B("B", 2, 2, B_data);

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
		B.print(stdout);
	}

	A -= B;

	if ( LOGGER(LL_VERBOSE) ) {
		A.print(stdout);
	}

	assert_matrix_value(A, A_data2, "A - B");
}

/**
 * Test matrix column subtraction.
 */
void test_subtract_columns()
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
	Matrix M("M", 3, 4, M_data1);
	Matrix a("a", 3, 1, a_data);;

	if ( LOGGER(LL_VERBOSE) ) {
		M.print(stdout);
		a.print(stdout);
	}

	M.subtract_columns(a);

	if ( LOGGER(LL_VERBOSE) ) {
		M.print(stdout);
	}

	assert_matrix_value(M, M_data2, "M - a * 1_N'");
}

/**
 * Test matrix row subtraction.
 */
void test_subtract_rows()
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
	Matrix M("M", 3, 4, M_data1);
	Matrix a("a", 1, 4, a_data);

	if ( LOGGER(LL_VERBOSE) ) {
		M.print(stdout);
		a.print(stdout);
	}

	M.subtract_rows(a);

	if ( LOGGER(LL_VERBOSE) ) {
		M.print(stdout);
	}

	assert_matrix_value(M, M_data2, "M - 1_N * a");
}

void print_usage()
{
	fprintf(stderr,
		"Usage: ./test-matrix [options]\n"
		"\n"
		"Options:\n"
		"  --loglevel LEVEL   set the log level (1=info, 2=verbose, 3=debug)\n"
	);
}

int main(int argc, char **argv)
{
	// parse command-line arguments
	struct option long_options[] = {
		{ "loglevel", required_argument, 0, 'e' },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case 'e':
			LOGLEVEL = (logger_level_t) atoi(optarg);
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

#ifdef __NVCC__
	magma_int_t stat = magma_init();
	assert(stat == MAGMA_SUCCESS);
#endif

	// run tests
	test_func_t tests[] = {
		test_identity,
		test_ones,
		test_random,
		test_zeros,
		test_copy,
		test_copy_columns,
		test_argmax,
		test_diagonalize,
		test_eigen,
		test_eigen2,
		test_inverse,
		test_mean_column,
		test_mean_row,
		test_norm,
		test_product,
		test_sum,
		test_transpose,
		test_add,
		test_subtract,
		test_assign_column,
		test_assign_row,
		test_elem_apply,
		test_elem_mult,
		test_subtract_columns,
		test_subtract_rows
	};
	int num_tests = sizeof(tests) / sizeof(test_func_t);

	int i;
	for ( i = 0; i < num_tests; i++ ) {
		test_func_t test = tests[i];

		printf("TEST %d\n\n", i + 1);

		test();
		putchar('\n');
	}

#ifdef __NVCC__
	stat = magma_finalize();
	assert(stat == MAGMA_SUCCESS);
#endif

	return 0;
}
