#include "matrixOperations.h"

/*  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GROUP 6 FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~  */
/*******************************************************************************
 * void matrix_eig(data_t *out_eig_vect, data_t*out_eig_vals, data_t* matrix, int rows, int cols);
 * Get eigenvalues and eigenvectors of symmetric matrix
 * NOTE: ONLY SYMMETRIC MATRICIES ATM
*******************************************************************************/
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors) {
/*	gsl_matrix * A = gsl_matrix_alloc (M->numRows, M->numCols);
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
*/



// ***********************************************************************************************************
// ***********************************************************************************************************
// going to use ddgev in BLAS to compute the eigenvalues/eigenvectors of the matrix
// The following is the routine:
// dggev (JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO)
// JOBVL    --->     'N' - do not compute left generalized eigenvector
//					 'V' - compute the left EV
// JOBVR    --->     'N' - do not compute right generalized eigenvector
//					 'V' - compute the right EV
// N        --->     INT The order of the matrices A, B, VL, and VR.  N >= 0.
// A        --->     A is DOUBLE PRECISION array, dimension (LDA, N)
//                   On entry, the matrix A in the pair (A,B).
//                   On exit, A has been overwritten.
// LDA      --->     INT The leading dimension of A
// B        --->     DOUBLE PRECISION array, dimensions (LDB, N)
//                   On entry, the matrix B in the pair (A, B)
//                   On exit, B is overwritten
// LDB      --->     INT The leading dimension of B
// ALPHAR   --->     DOUBLE PRECISION array, dimension (N)
// ALPHAI   --->     DOUBLE PRECISION array, dimension (N)
// BETA     --->     DOUBLE PRECISION array, dimension (N)
//                   On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
//                   be the generalized eigenvalues.  If ALPHAI(j) is zero, then
//                   the j-th eigenvalue is real; if positive, then the j-th and
//                   (j+1)-st eigenvalues are a complex conjugate pair, with
//                   ALPHAI(j+1) negative.
// VL       --->     VL is DOUBLE PRECISION array, dimension (LDVL,N)
//                   If JOBVL = 'V', the left eigenvectors u(j) are stored one
//                   after another in the columns of VL, in the same order as
//                   their eigenvalues. If the j-th eigenvalue is real, then
//                   u(j) = VL(:,j), the j-th column of VL. If the j-th and
//                   (j+1)-th eigenvalues form a complex conjugate pair, then
//                   u(j) = VL(:,j)+i*VL(:,j+1) and u(j+1) = VL(:,j)-i*VL(:,j+1).
//                   Each eigenvector is scaled so the largest component has
//                   abs(real part)+abs(imag. part)=1.
//                   Not referenced if JOBVL = 'N'.
// LDVL     --->     LDVL is INTEGER
//                   The leading dimension of the matrix VL. LDVL >= 1, and
//                   if JOBVL = 'V', LDVL >= N.
// VR       --->     VR is DOUBLE PRECISION array, dimension (LDVR,N)
//                   If JOBVR = 'V', the right eigenvectors v(j) are stored one
//                   after another in the columns of VR, in the same order as
//                   their eigenvalues. If the j-th eigenvalue is real, then
//                   v(j) = VR(:,j), the j-th column of VR. If the j-th and
//                   (j+1)-th eigenvalues form a complex conjugate pair, then
//                   v(j) = VR(:,j)+i*VR(:,j+1) and v(j+1) = VR(:,j)-i*VR(:,j+1).
//                   Each eigenvector is scaled so the largest component has
//                   abs(real part)+abs(imag. part)=1.
//                   Not referenced if JOBVR = 'N'.
// LDVR     --->     LDVR is INTEGER
//                   The leading dimension of the matrix VR. LDVR >= 1, and
//                   if JOBVR = 'V', LDVR >= N.
// WORK     --->     WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//                   On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
// LWORK    --->     LWORK is INTEGER
//                   The dimension of the array WORK.  LWORK >= max(1,8*N).
//                   For good performance, LWORK must generally be larger.
//                   If LWORK = -1, then a workspace query is assumed; the routine
//                   only calculates the optimal size of the WORK array, returns
//                   this value as the first entry of the WORK array, and no error
//                   message related to LWORK is issued by XERBLA.
// INFO     --->     INFO is INTEGER
//                   = 0:  successful exit
//                   < 0:  if INFO = -i, the i-th argument had an illegal value.
//                   = 1,...,N:
//                   The QZ iteration failed.  No eigenvectors have been
//                   calculated, but ALPHAR(j), ALPHAI(j), and BETA(j)
//                   should be correct for j=INFO+1,...,N.
//                   > N:  =N+1: other than QZ iteration failed in DHGEQZ.
//                   =N+2: error return from DTGEVC.
//

// ***********************************************************************************************************
// ***********************************************************************************************************
    dggev()
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
