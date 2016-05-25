// matrix test
#define XDIM 6
#define YDIM 6

#include <stdio.h>
#include <stdlib.h>
#include "matrixOperations.h"

int main (void) {
	
	FILE *output = fopen ("testResults.txt", "w");

	matrix_t *M = m_initialize (FILL, XDIM, YDIM);
    matrix_t *R; 

	fprintf (output, "M = \n");
	m_fprint (output, M);

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

    N = m_initialize (IDENTITY, XDIM, YDIM);
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

    // Causes Segfault
    //R = m_covariance (M);
	//fprintf (output, "m_covariance(M) = \n");
	//m_fprint (output, R);

    // Causes ABORT - Double Free?
    // m_free (R);
	// m_free (M);
	// m_free (R);

	return 0;
}

