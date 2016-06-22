/**
 * @file test_matrix.c
 *
 * Test suite for the matrix library.
 */
#include <stdio.h>
#include "matrix.h"

#define ROWS 6
#define COLS 6

/**
 * Helper function to fill a matrix with a constant value.
 */
void fill_matrix_constant(matrix_t *M, precision_t c)
{
	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = c;
		}
	}
}

/**
 * Helper function to fill a matrix with arbitrary data.
 */
void fill_matrix_data(matrix_t *M, precision_t data[][M->cols])
{
	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = data[i][j];
		}
	}
}

/**
 * Helper function to fill a matrix with an increasing value.
 */
void fill_matrix_linear(matrix_t *M)
{
	int i, j;
	for ( i = 0; i < M->rows; i++ ) {
		for ( j = 0; j < M->cols; j++ ) {
			elem(M, i, j) = j * M->rows + i;
		}
	}
}

int main (int argc, char **argv)
{
	matrix_t *M;

	matrix_t *A;
	matrix_t *B;

	// identity matrix
	M = m_identity(ROWS);

	printf("m_identity (%d) = \n", ROWS);
	m_fprint(stdout, M);

	m_free(M);

	// zero matrix
	M = m_zeros(ROWS, COLS);

	printf("m_zeros (%d, %d) = \n", ROWS, COLS);
	m_fprint(stdout, M);

	m_free(M);

	// COS, L1, L2 distance
	precision_t data1[][2] = {
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 0 }
	};

	M = m_zeros(3, 2);

	fill_matrix_data(M, data1);

	printf("M = \n");
	m_fprint(stdout, M);

	printf("d_COS(M[0], M[1]) = %lf\n", m_dist_COS(M, 0, M, 1));

	printf("d_L1(M[0], M[1]) = %lf\n", m_dist_L1(M, 0, M, 1));

	printf("d_L2(M[0], B[1]) = %lf\n", m_dist_L2(M, 0, M, 1));

	m_free(M);

	// eigenvalues and eigenvectors
	precision_t data2[][3] = {
		{ 2, 0, 0 },
		{ 0, 3, 4 },
		{ 0, 4, 9 }
	};

	matrix_t *M_eval;
	matrix_t *M_evec;

	M = m_initialize(3, 3);
	M_eval = m_initialize(3, 1);
	M_evec = m_initialize(3, 3);

	fill_matrix_data(M, data2);

	m_eigenvalues_eigenvectors(M, M_eval, M_evec);

	printf("M = \n");
	m_fprint(stdout, M);

	printf("eigenvalues of M = \n");
	m_fprint(stdout, M_eval);

	printf("eigenvectors of M = \n");
	m_fprint(stdout, M_evec);

	m_free(M);
	m_free(M_eval);
	m_free(M_evec);

	// matrix inverse
	matrix_t *M_inv;
	matrix_t *M_prod;

    M = m_identity(ROWS);
	M_inv = m_inverse(M);
	M_prod = m_matrix_multiply(M, M_inv);

    fprintf(stdout, "M = \n");
    m_fprint(stdout, M);
    fprintf(stdout, "M^-1 = \n");
    m_fprint(stdout, M_inv);
    fprintf(stdout, "M * M^-1 = \n");
    m_fprint(stdout, M_prod);

    m_free(M);
	m_free(M_inv);
	m_free(M_prod);

	precision_t data3[][3] = {
		{ 4, 1, 1 },
		{ 2, 1, -1 },
		{ 1, 1, 1 }
	};
	M = m_initialize(3, 3);
	fill_matrix_data(M, data3);

	M_inv = m_inverse(M);
	M_prod = m_matrix_multiply(M, M_inv);

    fprintf(stdout, "M = \n");
    m_fprint(stdout, M);
    fprintf(stdout, "M^-1 = \n");
    m_fprint(stdout, M_inv);
    fprintf(stdout, "M * M^-1 = \n");
    m_fprint(stdout, M_prod);

    m_free(M);
	m_free(M_inv);
	m_free(M_prod);

	// TODO: this test does not provide the correct inverse
	M = m_initialize(ROWS, ROWS);
	fill_matrix_linear(M);

	M_inv = m_inverse(M);
	M_prod = m_matrix_multiply(M, M_inv);

    fprintf(stdout, "M = \n");
    m_fprint(stdout, M);
    fprintf(stdout, "M^-1 = \n");
    m_fprint(stdout, M_inv);
    fprintf(stdout, "M * M^-1 = \n");
    m_fprint(stdout, M_prod);

    m_free(M);
	m_free(M_inv);
	m_free(M_prod);

	// matrix product
	A = m_initialize(ROWS, COLS + 2);
	B = m_initialize(COLS + 2, COLS + 1);

	fill_matrix_linear(A);
	fill_matrix_linear(B);

	M = m_matrix_multiply(A, B);

	printf("A = \n");
	m_fprint(stdout, A);

	printf("B = \n");
	m_fprint(stdout, B);

	printf("m_matrix_multiply (A, B) = \n");
	m_fprint(stdout, M);

	m_free(A);
	m_free(B);
	m_free(M);

	// mean column
	matrix_t *a;

	M = m_identity(ROWS);
	a = m_mean_column(M);

	printf("M = \n");
	m_fprint(stdout, M);

	printf("m_mean_column (M) = \n");
	m_fprint(stdout, a);

	m_free(M);
	m_free(a);

	// matrix transpose
	matrix_t *T;

	M = m_zeros(ROWS + 2, COLS);
	T = m_transpose(M);

	printf("M = \n");
	m_fprint(stdout, M);

	printf("m_transpose (M) = \n");
	m_fprint(stdout, T);

	m_free(M);
	m_free(T);

	// column normalization
	M = m_initialize(ROWS, COLS);
	a = m_initialize(ROWS, 1);

	fill_matrix_linear(M);
	fill_matrix_linear(a);

	printf("M = \n");
	m_fprint(stdout, M);

	printf("a = \n");
	m_fprint(stdout, a);

	m_normalize_columns(M, a);

	printf("m_normalize_columns (M, a) = \n");
	m_fprint(stdout, M);

	m_free(M);
	m_free(a);

/*
	// Test Group 2.0.0
	fprintf (output, "\n-------------Test Group 2.0.0 -------------\n");
	m_flipCols (M);
	fprintf (output, "m_flipCols(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_normalize (M);
	fprintf (output, "m_normalize(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_normalize (M);
    m_elem_mult(M, 35);
    fprintf (output, "m_normalize(M)\n");
	fprintf (output, "m_elem_mult(M, 35) =\n");
	m_fprint (output, M);
	m_free (M);

	// Test Group 2.0.1
	fprintf (output, "\n-------------Test Group 2.0.1 -------------\n");
	M = m_initialize (FILL, ROWS, COLS);
	m_elem_truncate (M);
	fprintf (output, "m_elem_truncate(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
    m_elem_divideByConst(M, 6);
	m_elem_truncate (M);
    fprintf (output, "m_divide_by_constant(M, 6)\n");
	fprintf (output, "m_elem_truncate(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_acos (M);
	fprintf (output, "m_elem_acos(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_sqrt (M);
	fprintf (output, "m_elem_sqrt(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_negate (M);
	fprintf (output, "m_elem_negate(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_exp (M);
	fprintf (output, "m_elem_exp(M) = \n");
	m_fprint (output, M);
	m_free (M);

	// Test Group 2.0.2
	fprintf (output, "\n-------------Test Group 2.0.2 -------------\n");
	precision_t x = 2.0;
	M = m_initialize (FILL, ROWS, COLS);
	m_elem_pow (M, x);
	fprintf (output, "m_elem_pow(M, x) = \n");
	m_fprint (output, M);
	m_free (M);


	M = m_initialize (FILL, ROWS, COLS);
	m_elem_mult (M, x);
	fprintf (output, "m_elem_mult(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_divideByConst (M, x);
	fprintf (output, "m_elem_divideByConst(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_divideByMatrix (M, x);
	fprintf (output, "m_elem_divideByMatrix(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, ROWS, COLS);
	m_elem_add (M, x);
	fprintf (output, "m_elem_add(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	// Test Group 2.1.0
	fprintf (output, "\n-------------Test Group 2.1.0 -------------\n");
	matrix_t *R;
	M = m_initialize (FILL, ROWS, COLS);

	R = m_sumCols (M);
	fprintf (output, "m_sumCols(M) = \n");
	m_fprint (output, R);
	m_free (R);

	R = m_meanCols (M);
	fprintf (output, "m_meanCols(M) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 2.1.1
	fprintf (output, "\n-------------Test Group 2.1.1 -------------\n");
	R = m_sumRows (M);
	fprintf (output, "m_sumRows(M) = \n");
	m_fprint (output, R);
	m_free (R);

	elem (M, 1, 2) = 0.0;  elem (M, 5, 5) = 0.0;
	elem (M, 4, 5) = 0.0;  elem (M, 5, 4) = 0.0;
	R = m_findNonZeros (M);
	fprintf (output, "m_findNonZeros(M) = \n");
	m_fprint (output, R);
	m_free (R);
	m_free (M);

	// Test Group 2.1.2
	fprintf (output, "\n-------------Test Group 2.1.2 -------------\n");
	matrix_t *A = m_initialize (FILL, ROWS, COLS);
	fprintf (output, "A = \n");
	m_fprint (output, A);

	R = m_reshape (A, ROWS / 2, COLS * 2);
	fprintf (output, "m_reshape (A, ROWS / 2, COLS * 2) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 3
	fprintf (output, "\n-------------Test Group 3 -------------\n");
	M = m_initialize (FILL, ROWS, COLS);

	R = m_sqrtm (M);
	fprintf (output, "m_sqrtm(M) = \n");
	m_fprint (output, R);
	m_free (R);


	precision_t val = m_determinant (M);
	fprintf (output, "m_determinant(M) = %lf\n", val);

	R = m_cofactor (M);
	fprintf (output, "m_cofactor(M) = \n");
	m_fprint (output, R);
	m_free (R);

    matrix_t *N = m_initialize (FILL, 3, 3);
	R = m_cofactor (N);
    fprintf (output, "Three-by-Three matrix N\n");
	fprintf (output, "m_cofactor(N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (FILL, 2, 2);
	R = m_cofactor (N);
    fprintf (output, "Two-by-Two matrix N\n");
	fprintf (output, "m_cofactor(N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (IDENTITY, ROWS, COLS);
	R = m_cofactor (N);
    fprintf (output, "Identity matrix 6x6 N\n");
    fprintf (output, "m_determinant(N)\n");
    fprintf (output, "%lf\n", m_determinant (N));
	fprintf (output, "m_cofactor(N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (IDENTITY, 5, 5);
	R = m_cofactor (N);
    fprintf (output, "Identity matrix 5x5 N\n");
    fprintf (output, "m_determinant(N)\n");
    fprintf (output, "%lf\n", m_determinant (N));
    fprintf (output, "m_cofactor(N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (IDENTITY, 4, 4);
	R = m_cofactor (N);
    fprintf (output, "Identity matrix 4x4 N\n");
    fprintf (output, "m_determinant (N)\n");
    fprintf (output, "%lf\n", m_determinant (N));
    fprintf (output, "m_cofactor (N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (IDENTITY, 3, 3);
	R = m_cofactor (N);
    fprintf (output, "Identity matrix 3x3 N\n");
    fprintf (output, "m_determinant (N)\n");
    fprintf (output, "%lf\n", m_determinant (N));
    fprintf (output, "m_cofactor (N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    N = m_initialize (IDENTITY, 2, 2);
	R = m_cofactor (N);
    fprintf (output, "Identity matrix 2x2 N\n");
    fprintf (output, "m_determinant (N)\n");
    fprintf (output, "%lf\n", m_determinant (N));
    fprintf (output, "m_cofactor (N) = \n");
	m_fprint (output, R);
	m_free (R);
    m_free (N);

    R = m_covariance (M);
	fprintf (output, "m_covariance(M) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 4
	fprintf (output, "\n-------------Test Group 4 -------------\n");
	matrix_t *B = m_initialize (FILL, ROWS, COLS);
	m_elem_add (B, -10.0);
	fprintf (output, "B =\n");
	m_fprint (output, B);

	R = m_dot_subtract (M, B);
	fprintf (output, "m_dot_subtract(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	R = m_dot_add (M, B);
	fprintf (output, "m_dot_add(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	R = m_dot_division (M, B);
	fprintf (output, "m_dot_division(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 5
	fprintf (output, "\n-------------Test Group 5 -------------\n");

	R = m_matrix_division(M, B);
	fprintf (output, "m_matrix_division(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	matrix_t *V = m_initialize (UNDEFINED, 1, 6);
	V->data[0] = 4; V->data[1] = 5; V->data[2] = 2; V->data[3] = 1;
	V->data[4] = 0; V->data[5] = 3;
	fprintf (output, "V = \n");
	m_fprint (output, V);

	R = m_reorder_columns (M, V);
	fprintf (output, "m_reorderCols (M, V) = \n");
	m_fprint (output, R);
	m_free (R);
*/

	return 0;
}
