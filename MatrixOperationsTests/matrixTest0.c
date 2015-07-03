#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"


int main (int argc, char **argv) {
	
	// Open the specified input and output files
	FILE *inputFile = fopen ("matrixTest_in.txt", "r");
	FILE *outputFile = fopen ("matrixTest_out.txt", "w");

	// Test subtractMatrix Column
	matrix_t *M = fscanMatrix(inputFile);
	matrix_t *V = fscanMatrix(inputFile);
	
	int i;
	for (i = 0; i < M->numCols; i++) {
		subtractMatrixColumn (M, i, V, 0);
	}
	
	fprintf (outputFile,	"------------------------------------------\n"
							"Subtraction result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, M);

	freeMatrix (V);
	freeMatrix (M);
	
	// Test matrixMultiply
	matrix_t *A = fscanMatrix(inputFile);
	matrix_t *B = fscanMatrix(inputFile);


	matrix_t *R1 = matrixMultiply (A, NOT_TRANSPOSED, B, NOT_TRANSPOSED, 0);

	fprintf (outputFile,	"------------------------------------------\n"
							"A x B result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, R1);

	matrix_t *R2 = matrixMultiply (A, TRANSPOSED, B, TRANSPOSED, 0);

	fprintf (outputFile,	"------------------------------------------\n"
							"A' x B' result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, R2);

	freeMatrix (R1);
	freeMatrix (R2);
	freeMatrix (B);

	R1 = matrixMultiply (A, NOT_TRANSPOSED, A, TRANSPOSED, 0);
	R2 = matrixMultiply (A, TRANSPOSED, A, NOT_TRANSPOSED, 0);

	fprintf (outputFile,	"------------------------------------------\n"
							"A x A' result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, R1);
	fprintf (outputFile,	"------------------------------------------\n"
							"A' x A result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, R2);

	freeMatrix (R1);
	freeMatrix (R2);

	// Test copyMatrix
	matrix_t *A_copy = copyMatrix (A);


	fprintf (outputFile,	"------------------------------------------\n"
							"A copy result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, A_copy);

	freeMatrix (A_copy);

	// Test calcSurrogateMatrix
	M = calcSurrogateMatrix (A);
	fprintf (outputFile,	"------------------------------------------\n"
							"calcSurrogateMatrix result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, M);

	freeMatrix (M);
	
	// Test calcMeanCol
	M = calcMeanCol (A);
	fprintf (outputFile,	"------------------------------------------\n"
							"calcMeanCol result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, M);

	freeMatrix (M);
	
	// Test calcEigenvectorsSymmetric
	M = matrixMultiply(A, TRANSPOSED, A, NOT_TRANSPOSED, 0);
	matrix_t *eigenvectors = calcEigenvectorsSymmetric (M);

	fprintf (outputFile,	"------------------------------------------\n"
							"eigenvectors of (A' x A) result\n"
							"------------------------------------------\n");
	fprintMatrix (outputFile, eigenvectors);

	freeMatrix (eigenvectors);
	freeMatrix (M);

	freeMatrix (A);
	return 0;
}

	
	

	
