/**
 * @file matrix_test.c
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
void fill_matrix(matrix_t *M, int c)
{
	int i, j;
	for ( i = 0; i < M->numRows; i++ ) {
		for ( j = 0; j < M->numCols; j++ ) {
			elem(M, i, j) = c;
		}
	}
}

int main (int argc, char **argv)
{
	FILE *output = fopen("test.log", "w");
	matrix_t *M;

	// identity matrix
	M = m_identity(ROWS);

	fprintf(output, "M = \n");
	m_fprint(output, M);

	m_free(M);

	// zero matrix
	M = m_zeros(ROWS, COLS);

	fprintf(output, "M = \n");
	m_fprint(output, M);

	m_free(M);

	// column normalization
	matrix_t *a;

	M = m_initialize(ROWS, COLS);
	a = m_initialize(ROWS, 1);

	fill_matrix(M, 5);
	fill_matrix(a, 1);

	fprintf(output, "M = \n");
	m_fprint(output, M);

	fprintf(output, "a = \n");
	m_fprint(output, a);

	m_normalize_columns(M, a);

	fprintf(output, "m_normalize_columns (M, a) = \n");
	m_fprint(output, M);

	m_free(M);
	m_free(a);

	// matrix transpose
	matrix_t *T;

	M = m_zeros(ROWS + 2, COLS);
	T = m_transpose(M);

	fprintf(output, "M = \n");
	m_fprint(output, M);

	fprintf(output, "m_transpose (M) = \n");
	m_fprint(output, T);

	m_free(M);
	m_free(T);

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

	M = m_initialize (FILL, ROWS, COLS);
	m_inverseMatrix (M);
	fprintf (output, "m_inverseMatrix(M) = \n");
	m_fprint (output, M);
	m_free (M);

    M = m_initialize (IDENTITY, ROWS, COLS);
    m_inverseMatrix (M);
    fprintf (output, "This test will take the inverse of the identity\n");
    fprintf (output, "m_inverseMatrix(M) = \n");
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
	precision x = 2.0;
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

	R = m_meanRows (M);
	fprintf (output, "m_meanRows(M) = \n");
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

	fprintf (output, "m_norm (M, specRow) is SKIPPED IN THIS TEST\n");

	R = m_sqrtm (M);
	fprintf (output, "m_sqrtm(M) = \n");
	m_fprint (output, R);
	m_free (R);


	precision val = m_determinant (M);
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

	R = m_matrix_multiply(M, B, 0);
	fprintf (output, "m_matrix_multiply(M, B, 0) = \n");
	m_fprint (output, R);
	m_free (R);

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

	// Test Group 6
	fprintf (output, "\n-------------Test Group 6 -------------\n");

	matrix_t *eigenvalues, *eigenvectors;
	m_eigenvalues_eigenvectors (M, &eigenvalues, &eigenvectors);
	fprintf (output, "M's eigenvalues =  \n");
	m_fprint (output, eigenvalues);

	fprintf (output, "M's eigenvectors = \n");
	m_fprint (output, eigenvectors);

	R = m_getSubMatrix (M, 2, 2, 3, 3);
	fprintf (output, "m_getSubMatrix (M, 2, 2, 3, 3) = \n");
	m_fprint (output, R);
*/

	fclose(output);

	return 0;
}
