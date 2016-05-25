// matrix test
//#define XDIM 6
//#define YDIM 6

#include <stdio.h>
#include <stdlib.h>
#include "matrixOperations.h"

int main(void)
{
    FILE *output = fopen ("testResults.txt", "w");

    // declare sizes for different test matrices
    int test1 = 5;
    matrix_t *M = m_initialize (FILL, test1, test1);
    matrix_t *sub = NULL;
    //matrix_t *eigenValues = m_initialize(FILL, test1, test1);
    //matrix_t *eigenVectors = m_initialize(FILL, test1, test1);

    // print starting matrix
	fprintf (output, "M = \n");
	m_fprint (output, M);

    // make a sub matrix
    sub = m_getSubMatrix(M, 0, 0, 2, 2);
    fprintf(output, "\nsub from [0,0] to [2,2] = \n");
    m_fprint(output, sub);




/***************************************************************************
    EIGENVALUE FUNCTION NOT WORKING BECAUSE OF GSL LIBRARY NOT BEING INCLUDED
    SO COMMENTING OUT THIS TEST
****************************************************************************
    // Test for eigenvalues/eigenvectors
    fprintf (output, "\n-------------Test Group 1: Size = 2 -------------\n");
    m_eigenvalues_eigenvectors (M, &eigenValues, &eigenVectors);
    fprintf (output, "m_eigenvalues_eigenvectors(M, eigenValues, eigenVectors) = \n");
    m_fprint(output, eigenValues);
    m_fprint(output, eigenVectors);
    m_free(M);
    m_free(eigenValues);
    m_free(eigenVectors);

    M = m_initialize (FILL, test2, test2);
    eigenValues = m_initialize(FILL, test2, test2);
    eigenVectors = m_initialize(FILL, test2, test2);

    fprintf (output, "\n-------------Test Group 1: Size = 2 -------------\n");
    m_eigenvalues_eigenvectors (M, &eigenValues, &eigenVectors);
    fprintf (output, "m_eigenvalues_eigenvectors(M, eigenValues, eigenVectors) = \n");
    m_fprint(output, eigenValues);
    m_fprint(output, eigenVectors);
    m_free(M);
    m_free(eigenValues);
    m_free(eigenVectors);

    M = m_initialize (FILL, test3, test3);
    eigenValues = m_initialize(FILL, test3, test3);
    eigenVectors = m_initialize(FILL, test3, test3);

    fprintf (output, "\n-------------Test Group 1: Size = 2 -------------\n");
    m_eigenvalues_eigenvectors (M, &eigenValues, &eigenVectors);
    fprintf (output, "m_eigenvalues_eigenvectors(M, eigenValues, eigenVectors) = \n");
    m_fprint(output, eigenValues);
    m_fprint(output, eigenVectors);
    m_free(M);
    m_free(eigenValues);
    m_free(eigenVectors);
    */

    return 0;

}
