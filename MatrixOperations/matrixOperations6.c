#include "matrixOperations6.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 6 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*******************************************************************************
 * void matrix_eig(data_t *out_eig_vect, data_t*out_eig_vals, data_t* matrix, int rows, int cols); 
 * Get eigenvalues and eigenvectors of symmetric matrix
 * NOTE: ONLY SYMMETRIC MATRICIES ATM
*******************************************************************************/
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors) {
	gsl_matrix * A = gsl_matrix_alloc (M->numRows, M->numCols);
	gsl_matrix * gslEigenvectors = gsl_matrix_alloc (M->numRows, M->numCols);
	gsl_vector * gslEigenvalues = gsl_vector_alloc (M->numRows);
	
	precision val;
	int i, j;
	// Copy M into A
	for (i = 0; i < M->numRows; i++) {
		for (j = 0; j < M->numCols; j++) {
			val = m_getelem(M, i, j);
			gsl_matrix_set (A, i, j, val);
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
	
	matrix_t *eigenvalues = m_initialize (UNDEFINED, gslEigenvalues->size, 1);
	matrix_t *eigenvectors = m_initialize (UNDEFINED, gslEigenvectors->size1, gslEigenvectors->size2);

	// Copy the eigenvalues into a column matrix
	for (i = 0; i < gslEigenvalues->size; i++) {
		val = gsl_vector_get (gslEigenvalues, i);
		m_setelem(val, eigenvalues, i, 0);
	}
	
	// Copy the eigenvectors into a regular matrix
	for (i = 0; i < gslEigenvectors->size1; i++) {
		for (j = 0; j < gslEigenvectors->size2; j++) {
			val = gsl_matrix_get (gslEigenvectors, i, j);
			m_setelem(val, eigenvectors, i, j);
		}
	}
	gsl_eigen_symmv_free (w);
	gsl_matrix_free (gslEigenvectors);
	gsl_matrix_free (A);
	gsl_vector_free (gslEigenvalues);
	
	*p_eigenvectors = eigenvectors;
	*p_eigenvalues = eigenvalues; 
    
}


/*******************************************************************************
 * void submatrix(data_t *outmatrix, data_t *matrix, int rows, int cols, int start_row, int start_col, int end_row, int end_col);
 * NOTE: THIS DIRECTLY MANIPULATES THE PARENTS DATA
*******************************************************************************/
matrix_t * m_getSubMatrix (matrix_t *M, int startRow, int startCol, int numRows, int numCols) {
	matrix_t *sub = (matrix_t *) malloc (sizeof (matrix_t));
	sub->numRows = numRows;
	sub->numCols = numCols;
	sub->span = M->span;
	sub->type = SUBMATRIX;
	sub->data = &(M->data[numRows * M->span + numCols]);

	return sub;
}


/*******************************************************************************
 * Helper functio just used for function below
*******************************************************************************/
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
void loadPPMtoMatrixCol (char *path, matrix_t *M, int specCol, unsigned char *pixels) {
	FILE *in = fopen (path, "r");
	char header[4];
	int height, width, size, i;
	int numPixels = M->numRows;
	precision intensity;
	
	fscanf (in, "%s", header);
	if (strcmp (header, "P3") == 0) {
		skip_to_next_value (in);
		fscanf (in, "%d", &height);
		skip_to_next_value (in);
		fscanf (in, "%d", &width);
		skip_to_next_value (in);
		fscanf (in, "%d", &size);
		skip_to_next_value (in);
		for (i = 0; i < numPixels * 3; i++) {
			fscanf(in, "%c", &pixels[i]);
		}
	} else if (strcmp (header, "P6") == 0){
		fscanf (in, "%d %d %d", &height, &width, &size);
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
		elem(M, i, specCol) = intensity;
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
void writePPMgrayscale (char * filename, matrix_t *M, int specCol, int height, int width) {

	int i;
	char c;

	assert (height * width == M->numRows); // Number of pixels must match
	FILE * out = fopen (filename, "w");

	// Write file header
	fprintf (out, "P6\n%d\n%d\n255\n", height, width);

	// Write pixel data
	for (i = 0; i < M->numRows; i++) {
		c = (char) elem(M, i, specCol);
		fputc (c, out);
		fputc (c, out);
		fputc (c, out);
	}
	fclose (out);
}


