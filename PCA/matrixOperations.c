
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include "matrix.h"

/*******************************************************************************
 * initializeMatrix
 *
 * Returns a matrix pointer to a matrix of size M x N
 *
 * Depending on the input variable "mode", data is either a 2D matrix of
 * 		1. undefined values
 * 		2. zeros
 * 
*******************************************************************************/
matrix_t * initializeMatrix (int mode, int numRows, int numCols) {
	
	uint64_t i;

	matrix_t *M = (matrix_t *) malloc (sizeof (matrix_t));
	M->numRows = numRows;
	M->numCols = numCols;
	if (mode == UNDEFINED) {
		//M->data = (precision *) malloc (numRows * numCols * sizeof (precision));
		M->data = (precision **) malloc (numRows * sizeof (precision *));
		for (i = 0; i < numRows; i++) {
			M->data[i] = (precision *) malloc (numCols * sizeof (precision));
			if (M->data[i] == NULL) {
				printf ("malloc failed when allocating %" PRIu64 "\n", i);
				assert (0);
			}
		}
	} else if (mode == ZEROS) {
		//M->data = (precision *) calloc (numRows * numCols, sizeof (precision));
		M->data = (precision **) malloc (numRows * sizeof (precision *));
		for (i = 0; i < numRows; i++) {
			M->data[i] = (precision *) calloc (numCols, sizeof (precision));
		}
	} else {
		printf ("initializeMatrix, Not valid mode\n");
		exit (5);
	}
	
	return M;
}


/*******************************************************************************
 * freeMatrix
 * 
 * Frees memory for matrix M
*******************************************************************************/
void freeMatrix (matrix_t *M) {
	uint64_t i;
	for (i = 0; i < M->numRows; i++) {
		free(M->data[i]);
	}
	free (M->data);
	free (M);
}


/*******************************************************************************
 * printMatrix
 * 
 * Prints matrix M to the stream specified
 * Prints numRows, numCols, then each whole row of the matrix (aka [0][0], [0][1]..)
 *
*******************************************************************************/
void fprintMatrix (FILE *stream, matrix_t *M) {

	uint64_t i, j;
	
	fprintf (stream, "%" PRIu64 " %" PRIu64 "\n", M->numRows, M->numCols);
	/*for (i = 0; i < M->numRows * M->numCols; i++) {
		fprintf (stream, "%lf ", M->data[i]);
		if ((i + 1) % M->numCols == 0) {
			fprintf (stream, "\n");
		}
	}*/
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			fprintf (stream, "%lf ", M->data[i][j]);
		}
		fprintf (stream, "\n");
	} 
}


/*******************************************************************************
 * fwriteMatrix
 * 
 * Writes matrix M to the stream specified
 * Writes numRows, numCols, then the data
*******************************************************************************/
void fwriteMatrix (FILE *stream, matrix_t *M) {
	uint64_t i;
	fwrite (&M->numRows, sizeof (uint64_t), 1, stream);
	fwrite (&M->numCols, sizeof (uint64_t), 1, stream);
	//fwrite (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	
	for (i = 0; i < M->numRows; i++) {
		fwrite (M->data[i], sizeof (precision), M->numCols, stream);
	} 
}


/*******************************************************************************
 * fscanMatrix
 *
 * Scans matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * fscanMatrix (FILE *stream) {

	uint64_t i, j, numRows, numCols;
	numRows = 0;
	numCols = 0;

	fscanf (stream, "%" PRIu64 " %" PRIu64 "", &numRows, &numCols);
	matrix_t *M = initializeMatrix(UNDEFINED, numRows, numCols);
	/*for (i = 0; i < numRows * numCols; i++) {
		fscanf (stream, "%lf", &(M->data[i]));
	}*/
	for (i = 0; i < numRows; i++) {
		for (j = 0; j < numCols; j++) {
			fscanf (stream, "%f", &(M->data[i][j]));
		}
	}

	return M;
}

/*******************************************************************************
 * freadMatrix
 *
 * reads matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * freadMatrix (FILE *stream) {
	uint64_t i, numRows, numCols;
	fread (&numRows, sizeof (uint64_t), 1, stream);
	fread (&numCols, sizeof (uint64_t), 1, stream);
	matrix_t *M = initializeMatrix (UNDEFINED, numRows, numCols);
	//fread (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	for (i = 0; i < numRows; i++) {
		fread (M->data[i], sizeof (precision), M->numCols, stream);
	}
	return M;
}


/*******************************************************************************
 * subtractMatrixColumn
 *
 * This function subtracts a column of M1 by another of M2. Which column this is
 * is specified by col1 and col2
*******************************************************************************/
void subtractMatrixColumn (matrix_t *M1, int col1, matrix_t *M2, int col2) {
	
	assert (M1->numRows == M2->numRows);
	assert (M1->numCols > col1);
	assert (M2->numCols > col2);

	uint64_t j;
	for (j = 0; j < M1->numRows; j++) {
		//M1->data[j * M1->numCols + col1] -= M2->data[j * M2->numCols + col2];
		M1->data[j][col1] -= M2->data[j][col2];
	}
}


/*******************************************************************************
 * dotProduct
 *
 * This function calculates the dot product of either
 *		1. 2 rows
 * 		2. 1 row and 1 col
 * 		2. 1 col and 1 row (yes its different in how its codes)
 * 		3. 2 cols
*******************************************************************************/
/*precision dotProduct (matrix_t *A,  int Apos, int Ao, matrix_t *B, int Bpos, int Bo) {

	uint64_t i;
	precision ans = 0;
	if (Ao == HORZ && Bo == HORZ) {
		assert (A->numCols == B->numCols);
		for (i = 0; i < A->numCols; i++) {
			ans += A->data[Apos][i] * B->data[Bpos][i];
		}
	} else if (Ao == HORZ && Bo == VERT) {
		assert (A->numCols == B->numRows);
		for (i = 0; i < A->numCols; i++) {
			ans += A->data[Apos][i] * B->data[i][Bpos];
		}
	} else if (Ao == VERT && Bo == HORZ) {
		assert (A->numRows == B->numCols);
		for (i = 0; i < A->numRows; i++) {
			ans += A->data[i][Apos] * B->data[Bpos][i];
		}
	} else if (Ao == VERT && Bo == VERT) {
		assert (A->numRows == B->numRows);
		for (i = 0; i < A->numRows; i++) {
			ans += A->data[i][Apos] * B->data[i][Bpos];
		}
	} else {
		printf ("Not a correct orientation for dot product");
		exit (60);
	}
	return ans;
}*/

/*******************************************************************************
 * Matrix multiply
 *
 * multiplies two matrices
 * 
 * Variables:
 * 		A - First matrix
 *		tranA - if A is transposed or not
 *		B - Second matrix
 *		tranB - if B is transposed or not
 * Output:
 * 		M - matrix that results from operation
 *
*******************************************************************************/
matrix_t * matrixMultiply (matrix_t *A, int tranA, matrix_t *B, int tranB, uint64_t maxCols) {
	
	uint64_t i, j, k, numCols;
	matrix_t *M;
	int progress = 0;
	if (tranA == NOT_TRANSPOSED && tranB == NOT_TRANSPOSED) {
		numCols = B->numCols;
		if (B->numCols != maxCols && maxCols != 0) {
			printf ("Matrix Size changed somewhere");
			numCols = maxCols;
		}
		M = initializeMatrix (ZEROS, A->numRows, numCols);
		for (i = 0; i < M->numRows; i++) {
			if ((int)(100.0*i/(float)M->numRows) >= progress){
				printf ("%d%%, ", progress);
				progress += 1;
				if (progress % 10 == 0) {
					printf ("\n");
				}
				fflush(stdout);
			}
			for (j = 0; j < M->numCols; j++) {
				for (k = 0; k < A->numCols; k++) {
					// M->data[i * M->numCols + j] += A->data[i * A->numCols + k] *
					//							   B->data[k * B->numCols + j];
					M->data[i][j] += A->data[i][k] * B->data[k][j];
				}
			}
		}
	} else if (tranA == TRANSPOSED && tranB == NOT_TRANSPOSED) {
		numCols = B->numCols;
		if (B->numCols != maxCols && maxCols != 0) {
			printf ("Matrix Size changed somewhere");
			numCols = maxCols;
		}
		M = initializeMatrix (ZEROS, A->numCols, numCols);
		for (i = 0; i < M->numRows; i++) {
			if ((int)(100.0*i/(float)M->numRows) >= progress){
				printf ("%d%%, ", progress);
				progress += 1;
				if (progress % 10 == 0) {
					printf ("\n");
				}
				fflush(stdout);
			}
			for (j = 0; j < M->numCols; j++) {
				for (k = 0; k < A->numRows; k++) {
					// M->data[i * M->numCols + j] += A->data[k * A->numCols + i] *
					// 							   B->data[k * B->numCols + j];
					M->data[i][j] += A->data[k][i] * B->data[k][j];
				}
			}
		}
	} else if (tranA == NOT_TRANSPOSED && tranB == TRANSPOSED) {
		numCols = B->numRows;
		if (B->numRows != maxCols && maxCols != 0) {
			printf ("Matrix Size changed somewhere");
			numCols = maxCols;
		}
		M = initializeMatrix (ZEROS, A->numRows, numCols);
		for (i = 0; i < M->numRows; i++) {
			if ((int)(100.0*i/(float)M->numRows) >= progress){
				printf ("%d%%, ", progress);
				progress += 1;
				if (progress % 10 == 0) {
					printf ("\n");
				}
				fflush(stdout);
			}
			for (j = 0; j < M->numCols; j++) {
				for (k = 0; k < A->numCols; k++) {
					// M->data[i * M->numCols + j] += A->data[i * A->numCols + k] *
					// 							   B->data[j * B->numCols + k];
					M->data[i][j] += A->data[i][k] * B->data[j][k];
				}
			}
		}
	} else {
		assert (tranA == TRANSPOSED && tranB == TRANSPOSED);
		numCols = B->numRows;
		if (B->numRows != maxCols && maxCols != 0) {
			printf ("Matrix Size changed somewhere");
			numCols = maxCols;
		}
		M = initializeMatrix (ZEROS, A->numCols, numCols);
		for (i = 0; i < M->numRows; i++) {
			if ((int)(100.0*i/(float)M->numRows) >= progress){
				printf ("%d%%, ", progress);
				progress += 1;
				if (progress % 10 == 0) {
					printf ("\n");
				}
				fflush(stdout);
			}
			for (j = 0; j < M->numCols; j++) {
				for (k = 0; k < A->numRows; k++) {
					// M->data[i * M->numCols + j] += A->data[k * A->numCols + i] * 
					//							   B->data[j * B->numCols + k];
					
					M->data[i][j] += A->data[k][i] * B->data[j][k];
				}
			}
		}
	} 
	
	return M;
}
				
/*******************************************************************************
 * copyMatrix
 *
 * Copies matrix M into a new matrix
 *
*******************************************************************************/
matrix_t * copyMatrix (matrix_t *M) {
	uint64_t i;

	matrix_t *C = (matrix_t *) malloc (sizeof(matrix_t));
	C->numRows = M->numRows;
	C->numCols = M->numCols;
	
	// C->data = (precision *) malloc (C->numRows * C->numCols * sizeof (precision));
	C->data = (precision **) malloc (C->numRows * sizeof (precision *));
	// memcpy(C->data, M->data, C->numRows * C->numCols * sizeof (precision));
	for (i = 0; i < C->numRows; i++) {
		C->data[i] = malloc (C->numCols * sizeof (precision));
		memcpy (C->data[i], M->data[i], C->numCols *sizeof (precision));
	}
	
	
	return C;
}


/*******************************************************************************
 * calcEigenvectorsSymmetric
 * 
 * This calculates the eigenvectors of matrix M
 * This uses the gsl library to perform the eigenvector calculations
 * 
 * NOTE: M must be symmetric
*******************************************************************************/
matrix_t * calcEigenvectorsSymmetric (matrix_t *M) {

	assert (M->numRows == M->numCols);

	uint64_t i, j;

	gsl_matrix * A;
	gsl_matrix * gslEigenvectors = gsl_matrix_alloc (M->numRows, M->numCols);
	gsl_vector * gslEigenvalues = gsl_vector_alloc (M->numRows);


	// Create gsl matrix
	A = gsl_matrix_alloc (M->numRows, M->numCols);
	
	// Copy M into A
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			//gsl_matrix_set (A, i, j, M->data[i * M->numCols + j]);
			gsl_matrix_set (A, i, j, M->data[i][j]);
		}
	}

	// Compute the Eigenvalues using the GSL library
	// Allocate workspace
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (M->numRows);

	gsl_eigen_symmv (A, gslEigenvalues, gslEigenvectors, w);

	// ********************************************************
	// COMMENT
	// We might need to normalize the eigenvectors here or something
	// to match matlab eigenvectors, they don't HAVE to to match but
	// its at least something to keep in mind
	// ********************************************************
	
	matrix_t *eigenvectors = initializeMatrix (UNDEFINED, gslEigenvectors->size1, gslEigenvectors->size2);

	// Copy the eigenvectors into a regular matrix
	for (i = 0; i < gslEigenvectors->size1; i++) {
		for (j = 0; j < gslEigenvectors->size2; j++) {
			// eigenvectors->data[i * eigenvectors->numCols + j] =
			//			gsl_matrix_get (gslEigenvectors, i, j);
			eigenvectors->data[i][j] = gsl_matrix_get (gslEigenvectors, i, j);
		}
	}
	//assert (gslEigenvectors->size1 == eigenvectors->size
	gsl_eigen_symmv_free (w);
	gsl_matrix_free (gslEigenvectors);
	gsl_matrix_free (A);
	gsl_vector_free (gslEigenvalues);
	//matrix_t *eigenvectors = initializeMatrix (ZEROS, M->numRows, M->numCols);
	return eigenvectors;
	
}


/*******************************************************************************
 * calcSurrogateMatrix
 *
 * Calculates the multiplication of A' * A
 * 
 * This exploits that this will be a symmetric
 * matrix to do half the calculations so it is 
 * faster than the usual matrixMultiply
*******************************************************************************/
matrix_t * calcSurrogateMatrix (matrix_t *A) {

	uint64_t i, j, k;

	matrix_t *L = initializeMatrix (ZEROS, A->numCols, A->numCols);
	int progress = 0;
	for (i = 0; i < A->numCols; i++) {
		if ((int)(100.0*i/(float)A->numCols) >= progress){
			printf ("%d%%, ", progress);
			progress += 1;
			if (progress % 10 == 0) {
				printf ("\n");
			}
			fflush(stdout);
		}
		for (j = i; j < A->numCols; j++) {
			for (k = 0; k < A->numRows; k++) {
				// L->data[i * L->numCols + j] += A->data[k * L->numCols + i] *
				//							   A->data[k * L->numCols + j];
				L->data[i][j] += A->data[k][i] * A->data[k][j];
			}
			//L->data[j * L->numCols + i] = L->data[i * L->numCols + j];
			L->data[j][i] = L->data[i][j];
		}
	}
	
	return L;
}


void skip_to_next_value(FILE* in)
{
   char ch = fgetc(in);
   while(ch == '#' || isspace(ch))
   {
       if(ch == '#')
       {
          while(ch != '\n') 
          {
             ch = fgetc(in);
          }      
       }
       else
       {
          while(isspace(ch))
          {
             ch = fgetc(in);             
          }
       }
   }
   
   ungetc(ch,in); //return last read value
}

/*******************************************************************************
 * loadPPMtoMatrixCol
 * 
 * This function loads the pixel data of a PPM image as a single column vector
 * in the preinitialized matrix M. It will load it into the column specified as
 * the specCol parameter. 
 * 
 * This function automatically turns any picture to grayscale if it is not
 * already
 * NOTE : currently this is set manually with the #define IS_COLOR in matrix.h
 *
 * NOTE : pixels is a matrix that must be allocated beforehand. This is to speed
 * up execution time if this function is called multiple times on the same size
 * image as it doesn't have to malloc and free that array every time.
*******************************************************************************/
void loadPPMtoMatrixCol (char *path, matrix_t *M, uint64_t specCol, unsigned char *pixels) {
	FILE *in = fopen (path, "r");
	char header[4];
	uint64_t height, width, size, i;
	uint64_t numPixels = M->numRows;
	precision intensity;
	
	fscanf (in, "%s", header);
	if (strcmp (header, "P3") == 0) {
		skip_to_next_value (in);
		fscanf (in, "%" PRIu64 "", &height);
		skip_to_next_value (in);
		fscanf (in, "%" PRIu64 "", &width);
		skip_to_next_value (in);
		fscanf (in, "%" PRIu64 "", &size);
		skip_to_next_value (in);
		for (i = 0; i < numPixels * 3; i++) {
			fscanf(in, "%c", &pixels[i]);
		}
	} else if (strcmp (header, "P6") == 0){
		fscanf (in, "%" PRIu64 " %" PRIu64 " %" PRIu64 "", &height, &width, &size);
		skip_to_next_value(in);
		fread (pixels, 3 * sizeof (unsigned char), numPixels, in);
	} else {
		printf ("Error not a P3 or P6 PPM");
		exit (8);
	}

	for (i = 0; i < numPixels; i++) {
		intensity = 0.299 * (precision)pixels[3*i] +
					0.587 * (precision)pixels[3*i+1] +
					0.114 * (precision) pixels[3*i+2];
		//M->data[i * M->numCols + specCol] = intensity;
		M->data[i][specCol] = intensity;
	}
	
	fclose (in);
}


/*******************************************************************************
 * writePPMgrayscale
 *
 * This writes a column vector of M (column specified by specCol) as a
 * grayscale ppm image. The height and width of the image must be specified 
 *
*******************************************************************************/
void writePPMgrayscale (char * filename, matrix_t *M, uint64_t specCol, uint64_t height, uint64_t width) {

	uint64_t i;
	char c;

	assert (height * width == M->numRows); // Number of pixels must match
	FILE * out = fopen (filename, "w");

	// Write file header
	fprintf (out, "P6\n%" PRIu64 "\n%" PRIu64 "\n255\n", height, width);

	// Write pixel data
	for (i = 0; i < M->numRows; i++) {
		//c = (char) M->data[i * M->numCols + specCol];
		c = (char) M->data[i][specCol];
		fputc (c, out);
		fputc (c, out);
		fputc (c, out);
	}
	fclose (out);
}


/*******************************************************************************
 * calcMeanCol
 *
 * This calculates the mean column in a matrix, or the average of each row
 * of the matrix
 *
*******************************************************************************/
matrix_t * calcMeanCol (matrix_t *A) {

	uint64_t i, j;
	matrix_t *m = initializeMatrix (ZEROS, A->numRows, 1);
	
	for (i = 0; i < A->numCols; i++) {
		for (j = 0; j < A->numRows; j++) {
			//m->data[j] += A->data[j* A->numCols + i] / (precision) A->numCols;
			m->data[j][0] += A->data[j][i] / (precision) A->numCols;
			// NOTE: could be losing precision here
		}
	}
	
	return m;
}


/*******************************************************************************
 * transpose matrix
 *
 * This function transposes a matrix
 *
*******************************************************************************/
matrix_t * transposeMatrix (matrix_t *M) {
	uint64_t i, j;

	matrix_t *T = initializeMatrix (UNDEFINED, M->numCols, M->numRows);
	
	for (i = 0; i < T->numRows; i++) {
		for (j = 0; j < T->numCols; j++) {
			T->data[i][j] = M->data[j][i];
		}
	}
	return T;
}
