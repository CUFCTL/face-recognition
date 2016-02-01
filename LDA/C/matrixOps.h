

#define precision double
#define UNDEFINED 0
#define ZEROS 1
#define ONES 2
#define FILL 3
#define IDENTITY 4
#define NOT_TRANSPOSED 0
#define TRANSPOSED 1
#define HORZ 0
#define VERT 1
#define COLOR 0
#define GRAYSCALE 1
#define PARENT 0
#define SUBMATRIX 1
#define IS_COLOR GRAYSCALE


typedef struct {
	precision *data;
	int numRows;
	int numCols;
	int span;
	int type;
} matrix_t;


// Group 0 - retreive or set one value in a matrix
void m_setElem (precision val, matrix_t *M, int i, int j);
precision m_getElem (matrix_t *M, int i, int j);

// Group 1 - initialization, input, output, and copy ops
matrix_t * m_initialize (int mode, int numRows, int numCols);
void m_free (matrix_t *M);
void m_fprint (FILE *stream, matrix_t *M);
void m_fwrite (FILE *stream, matrix_t *M);
matrix_t * m_fscan (FILE *stream);
matrix_t * m_fread (FILE *stream);
matrix_t * m_copy (matrix_t *M);

/***************** Group 2 - Operations on a single matrix *******************/
/***** 2.0 - No return values, operate directly on M's data *****/
// 2.0.0
//	- Not element wise operation
//	- no extra inputs
void m_flipCols (matrix_t *M);
void m_normalize (matrix_t *M);
void m_inverseMatrix (matrix_t *M); // Must be square matrix
// 2.0.1
//	- element wise math operation
//	- no extra inputs
void m_elem_truncate (matrix_t *M);
void m_elem_acos (matrix_t *M);
void m_elem_sqrt (matrix_t *M);
void m_elem_negate (matrix_t *M);
void m_elem_exp (matrix_t *M);
// 2.0.2
//	- element wise math operation
//	- has a second input operation relies on
void m_elem_pow (matrix_t *M, precision x);
void m_elem_mult (matrix_t *M, precision x);
void m_elem_divideByConst (matrix_t *M, precision x);
void m_elem_divideByMatrix (matrix_t *M, precision x);
void m_elem_add (matrix_t *M, precision x);

/***** 2.1 - returns a matrix, does not change input matrix M *****/
/***** No other inputs, except for m_reshape *****/
// 2.1.0
//	- returns row vector
matrix_t * m_sumCols (matrix_t *M);
matrix_t * m_meanCols (matrix_t *M);
// 2.1.1
//	- returns column vector
matrix_t * m_sumRows (matrix_t *M);
matrix_t * m_meanRows (matrix_t *M);
matrix_t * m_findNonZeros (matrix_t *M);
// 2.1.2
//	- reshapes data in matrix to new form
matrix_t * m_transpose (matrix_t *M);
matrix_t * m_reshape (matrix_t *M, int newNumRows, int newNumCols);

// Group 3 - complex linear algebra functions of a single matrix
precision m_norm (matrix_t *M, int specRow);
matrix_t * m_sqrtm (matrix_t *M);
precision m_determinant (matrix_t *M);
matrix_t * m_cofactor (matrix_t *M);
matrix_t * m_covariance (matrix_t *M);

// Group 4 - ops with 2 matrices that return a matrix of same size
matrix_t * m_dot_subtract (matrix_t *A, matrix_t *B);
matrix_t * m_dot_add (matrix_t *A, matrix_t *B);
matrix_t * m_dot_division (matrix_t *A, matrix_t *B);

// Group 5 - ops with 2 matrices that return a matrix of diff size
matrix_t * m_matrix_multiply (matrix_t *A, matrix_t *B, int maxCols);
matrix_t * m_matrix_division (matrix_t *A, matrix_t *B);
matrix_t * m_reorder_columns (matrix_t *M, matrix_t *V);

// Group 6 - other, doesn;t really fit in anywhere
void m_eigenvalues_eigenvectors (matrix_t *M, matrix_t **p_eigenvalues, matrix_t **p_eigenvectors);
matrix_t * m_getSubMatrix (matrix_t *M, int startRow, int startCol, int numRows, int numCols);

