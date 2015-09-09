#include "matrixOperations.h"

// Group 1 - initialization, input, output, and copy ops
matrix_t * m_initialize (int mode, int numRows, int numCols);
void m_free (matrix_t *M);
void m_fprint (FILE *stream, matrix_t *M);
void m_fwrite (FILE *stream, matrix_t *M);
matrix_t * m_fscan (FILE *stream);
matrix_t * m_fread (FILE *stream);
matrix_t * m_copy (matrix_t *M);

