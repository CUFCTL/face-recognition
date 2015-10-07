// matrix test
#define XDIM 6
#define YDIM 6

#include <stdio.h>
#include <stdlib.h>
#include "matrixOperations.h"


int main (void) {

	FILE *output = fopen ("testResults.txt", "w");

	matrix_t *M = m_initialize (FILL, XDIM, YDIM);
	matrix_t *B = m_initialize (FILL, XDIM, YDIM);
	matrix_t *R = m_initialize (FILL, XDIM, YDIM);

	fprintf (output, "M = \n");
	m_fprint (output, M);


	// Test Group 5
	fprintf (output, "\n-------------Test Group 5 -------------\n");
	fprintf (output, "\n-------------  Multiply  -------------\n");
	R = m_matrix_multiply(M, B, 0);
	fprintf (output, "m_matrix_multiply(M, B, 0) = \n");
	m_fprint (output, R);
	m_free (R);

	fprintf (output, "\n-------------  Division  -------------\n");
	R = m_matrix_division(M, B);
	fprintf (output, "m_matrix_division(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	matrix_t *V = m_initialize (UNDEFINED, 1, 6);
	V->data[0] = 4; V->data[1] = 5; V->data[2] = 2; V->data[3] = 1;
	V->data[4] = 0; V->data[5] = 3;
	fprintf (output, "V = \n");
	m_fprint (output, V);

	fprintf (output, "\n-----------  Reorder Coumns  -----------\n");
	R = m_reorder_columns (M, V);
	fprintf (output, "m_reorderCols (M, V) = \n");
	m_fprint (output, R);
	m_free (R);

	//m_free (eigenvectors);
	//m_free (eigenvalues);
	m_free (M);
	m_free (B);

	return 0;
}
