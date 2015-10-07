// matrix test
#define XDIM 6
#define YDIM 6

#include <stdio.h>
#include <stdlib.h>
#include "matrixOperations.h"


int main (void) {

	FILE *output = fopen ("testResults.txt", "w");

	matrix_t *M = m_initialize (FILL, XDIM, YDIM);
	//matrix_t *B = m_initialize (FILL, XDIM, YDIM);
	//matrix_t *R = m_initialize (FILL, XDIM, YDIM);

	//fprintf (output, "M = \n");
	//m_fprint (output, M);

	m_free (M);

	return 0;
}
