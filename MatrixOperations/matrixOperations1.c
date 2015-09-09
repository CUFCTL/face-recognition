#include "matrixOperations1.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 1 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
//  initialization, free, input, output, and copy functions
/*******************************************************************************
 * m_initialize
 *
 * Returns a matrix pointer to a matrix of size M x N
 *
 * Depending on the input variable "mode", data is either a 2D matrix of
 * 		1. ZEROS = zeros
 *		2. IDENTITY = identity matrix
 *		3. UNDEFINED = undefined values
 * 		4. ONES = all ones
 *		5. FILL = each element increases by one
 *
 * ICA:
 *		void allocate_matrix(data_t **vector, int rows, int cols);
 *		void allocate_vector(data_t **vector, int length);
 *		void ones(data_t *onesMat, int rows, int cols);
 * 		void eye(data_t *identity, int dimension);
 * 		void fill_matrix(data_t *matrix, int rows, int cols);
*******************************************************************************/
matrix_t * m_initialize (int mode, int numRows, int numCols) {
	int i, j;
	matrix_t *M = (matrix_t *) malloc (sizeof (matrix_t));
	M->numRows = numRows;
	M->numCols = numCols;
	M->span = numCols;
	M->type = PARENT; // not submatrix
	if (mode == ZEROS || mode == IDENTITY) {
		M->data = (precision *) calloc (numRows * numCols, sizeof (precision));
		if (mode == IDENTITY) {
			assert (numRows == numCols);
			for (i = 0; i < numRows; i++) {
                elem(M, i, i) = 1;	
			}
		}
	} else if (mode == UNDEFINED || mode == ONES || mode == FILL){
		M->data = (precision *) malloc (numRows * numCols * sizeof (precision));
		if (mode == ONES) {
			for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
				    elem(M, i, j) = 1.0;
                }
			}
		} else if (mode == FILL) {
			for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
				    elem(M, i, j) = i * numRows + j;
                }
			}
		}
	} else {
		printf ("m_initialize, Not valid mode\n");
		exit (5);
	}
	
	return M;
}


/*******************************************************************************
 * m_free
 * 
 * Frees memory for matrix M
 * ICA:
 *		void free_matrix(data_t **matrix);
 *		void free_vector(data_t **vector);
*******************************************************************************/
void m_free (matrix_t *M) {
	if (M->type != SUBMATRIX) {
		free (M->data);
	}
	free (M);
}


/*******************************************************************************
 * m_fprint
 * 
 * Prints matrix M to the stream specified
 * Prints numRows, numCols, then each whole row of the matrix (aka [0][0], [0][1]..)
 *
 * ICA:
 *		void print_matrix(data_t *matrix, int rows, int cols);
*******************************************************************************/
void m_fprint (FILE *stream, matrix_t *M) {

	int i, j;
	
	fprintf (stream, "%d %d\n", M->numRows, M->numCols);
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			fprintf (stream, "%lf ", elem(M, i, j));
		}
		fprintf (stream, "\n");
	}
	fflush (stream);
}


/*******************************************************************************
 * m_fwrite
 * 
 * Writes matrix M to the stream specified
 * Writes numRows, numCols, then the data
 * NOTE: will not work with submatrixes right now
*******************************************************************************/
void m_fwrite (FILE *stream, matrix_t *M) {
	fwrite (&M->numRows, sizeof (unsigned long int), 1, stream);
	fwrite (&M->numCols, sizeof (unsigned long int), 1, stream);
	if (M->type == PARENT) {
		fwrite (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	} else {
		int i;
		for (i = 0; M->numRows; i++) {
			fwrite (&(M->data[i * M->span]), sizeof (precision), M->numCols, stream);
		}
	}
}


/*******************************************************************************
 * m_fscan
 *
 * Scans matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * m_fscan (FILE *stream) {

	int i, j, numRows, numCols;
	numRows = 0;
	numCols = 0;	

	fscanf (stream, "%d %d", &numRows, &numCols);
	matrix_t *M = m_initialize(UNDEFINED, numRows, numCols);
	for (i = 0; i < numRows; i++) {
		for (j = 0; j < numCols; j++) {
			fscanf (stream, "%lf", &(elem(M, i, j)));
		}
	}

	return M;
}

/*******************************************************************************
 * m_fread
 *
 * reads matrix written by printMatrix in stream specified
*******************************************************************************/
matrix_t * m_fread (FILE *stream) {
	int numRows, numCols;
	fread (&numRows, sizeof (unsigned long int), 1, stream);
	fread (&numCols, sizeof (unsigned long int), 1, stream);
	matrix_t *M = m_initialize (UNDEFINED, numRows, numCols);
	fread (M->data, sizeof (precision), M->numRows * M->numCols, stream);
	return M;
}


/*******************************************************************************
 * m_copy
 *
 * Copies matrix M into a new matrix
 *
 * ICA:
 * 		data_t* copy(data_t* orig,int rows,int cols);
*******************************************************************************/
matrix_t * m_copy (matrix_t *M) {
	int i, j;
	
	matrix_t *C = (matrix_t *) malloc (sizeof(matrix_t));
	C->numRows = M->numRows;
	C->numCols = M->numCols;
	
	C->data = (precision *) malloc (C->numRows * C->numCols * sizeof (precision));
	if (M->numCols == M->span) {
		memcpy(C->data, M->data, C->numRows * C->numCols * sizeof (precision));
	} else {
		for (i = 0; i < C->numRows; i++) {
			for (j = 0; j < C->numCols; j++) {
                elem(C, i, j) = elem(M, i, j);
            }
		}
	}
	return C;
}





