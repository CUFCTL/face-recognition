// matrix test
#define XDIM 6
#define YDIM 6

#include <stdio.h>
#include <stdlib.h>
#include "matrixOps.h"

int main (void) {
	
	FILE *output = fopen ("testResults.txt", "w");

	matrix_t *M = m_initialize (FILL, XDIM, YDIM);

	fprintf (output, "M = \n");
	m_fprint (output, M);

	// Test Group 2.0.0
	fprintf (output, "\n-------------Test Group 2.0.0 -------------\n");
	m_flipCols (M);
	fprintf (output, "m_flipCols(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_normalize (M);
	fprintf (output, "m_normalize(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_inverseMatrix (M);
	fprintf (output, "m_inverseMatrix(M) = \n");
	m_fprint (output, M);
	m_free (M);
	
	// Test Group 2.0.1
	fprintf (output, "\n-------------Test Group 2.0.1 -------------\n");
	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_truncate (M);
	fprintf (output, "m_elem_truncate(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_acos (M);
	fprintf (output, "m_elem_acos(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_sqrt (M);
	fprintf (output, "m_elem_sqrt(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_negate (M);
	fprintf (output, "m_elem_negate(M) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_exp (M);
	fprintf (output, "m_elem_exp(M) = \n");
	m_fprint (output, M);
	m_free (M);

	// Test Group 2.0.2
	fprintf (output, "\n-------------Test Group 2.0.2 -------------\n");
	precision x = 2.0;
	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_pow (M, x);
	fprintf (output, "m_elem_pow(M, x) = \n");
	m_fprint (output, M);
	m_free (M);


	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_mult (M, x);
	fprintf (output, "m_elem_mult(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_divideByConst (M, x);
	fprintf (output, "m_elem_divideByConst(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_divideByMatrix (M, x);
	fprintf (output, "m_elem_divideByMatrix(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	M = m_initialize (FILL, XDIM, YDIM);
	m_elem_add (M, x);
	fprintf (output, "m_elem_add(M, x) = \n");
	m_fprint (output, M);
	m_free (M);

	// Test Group 2.1.0
	fprintf (output, "\n-------------Test Group 2.1.0 -------------\n");
	matrix_t *R;
	M = m_initialize (FILL, XDIM, YDIM);
	
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

	m_setElem (0.0, M, 1, 2);  m_setElem (0.0, M, 5, 5);
	m_setElem (0.0, M, 4, 5);  m_setElem (0.0, M, 5, 4);
	R = m_findNonZeros (M);
	fprintf (output, "m_findNonZeros(M) = \n");
	m_fprint (output, R);
	m_free (R);
	m_free (M);

	// Test Group 2.1.2
	fprintf (output, "\n-------------Test Group 2.1.2 -------------\n");
	matrix_t *A = m_initialize (FILL, XDIM, YDIM);
	fprintf (output, "A = \n");
	m_fprint (output, A);
	
	R = m_transpose (A);
	fprintf (output, "m_transpose (A) = \n");
	m_fprint (output, R);
	m_free (R);

	R = m_reshape (A, XDIM / 2, YDIM * 2);
	fprintf (output, "m_reshape (A, XDIM / 2, YDIM * 2) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 3
	fprintf (output, "\n-------------Test Group 3 -------------\n");
	M = m_initialize (FILL, XDIM, YDIM);
	
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

	R = m_covariance (M);
	fprintf (output, "m_covariance(M) = \n");
	m_fprint (output, R);
	m_free (R);

	// Test Group 4
	fprintf (output, "\n-------------Test Group 4 -------------\n");
	matrix_t *B = m_initialize (FILL, XDIM, YDIM);
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

	// Test Group 7
	fprintf (output, "\n GROUP 7 IS NOT TESTED IN THIS TEST\n");

	m_free (eigenvectors);
	m_free (eigenvalues);
	m_free (M);
	m_free (B);
	m_free (A);
	m_free (R);

	return 0;
}

