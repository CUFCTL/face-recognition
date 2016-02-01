// MATRIX datatype unit test

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "matrix.h"

int main()
{
    MATRIX *A = matrix_constructor(5, 5);
    assert(A != NULL); printf("pointer passed\n");
    assert(A->rows == 5); printf("rows passed\n");
    assert(A->cols == 5); printf("cols passed\n");
    assert(A->data != NULL); printf("data pointer passed\n");
    assert(A->data[0][1] == A->data[1][0]); printf("allocation passed\n");

    matrix_destructor(A);

    return 0;
}
