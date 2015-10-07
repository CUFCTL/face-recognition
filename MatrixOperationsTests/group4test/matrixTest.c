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

	// Test Group 4
	fprintf (output, "\n-------------Test Group 4 -------------\n");

	fprintf (output, "\n-------------subtraction-------------\n");
	fprintf(output, "\n\ntesting _dot_subtract using M and B to equal R\n");
	R = m_dot_subtract (M, B);
	fprintf (output, "m_dot_subtract(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	fprintf (output, "\n-------------addition-------------\n");
	fprintf(output, "\n\ntesting m_dot_add using matrices M and B to equal R");
	R = m_dot_add (M, B);
	fprintf (output, "\nm_dot_add(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	fprintf (output, "\n-------------division-------------\n");
	fprintf(output, "\n\ntesting m_dot_division using M and B to equal R");
	R = m_dot_division (M, B);
	fprintf (output, "\nm_dot_division(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	m_free(B);
	m_free(M);

	M = m_initialize (FILL, 2, 4);
	B = m_initialize (FILL, 2, 4);
	R = m_initialize (FILL, 2, 4);

	m_fprint(output, M);
	fprintf(output, "\nNOW USING A 2X4 MATRIX TO SEE IF THINGS STILL WORK");
	fprintf (output, "\n-------------subtraction-------------\n");
	fprintf(output, "\n\ntesting _dot_subtract using M and B to equal R\n");
	R = m_dot_subtract (M, B);
	fprintf (output, "m_dot_subtract(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	fprintf (output, "\n-------------addition-------------\n");
	fprintf(output, "\n\ntesting m_dot_add using matrices M and B to equal R");
	R = m_dot_add (M, B);
	fprintf (output, "\nm_dot_add(M, B) = \n");
	m_fprint (output, R);
	m_free (R);

	fprintf (output, "\n-------------division-------------\n");
	fprintf(output, "\n\ntesting m_dot_division using M and B to equal R");
	R = m_dot_division (M, B);
	fprintf (output, "\nm_dot_division(M, B) = \n");
	m_fprint (output, R);
	m_free (R);


	// Test Group 6
    /*	fprintf (output, "\n-------------Test Group 6 -------------\n");

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
	// Test Group 7
	fprintf (output, "\n GROUP 7 IS NOT TESTED IN THIS TEST\n");

	//m_free (eigenvectors);
	//m_free (eigenvalues);
	m_free (M);
	m_free (B);

	return 0;
}
