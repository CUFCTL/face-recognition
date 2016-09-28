/**
 * @file fpica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
#include "database.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#define maxNumIterations 1000
#define epsilon 0.0001

void randomize_vector(matrix_t * m);
precision_t norm(matrix_t * m);
void m_copy_vector_into_column(matrix_t * M, matrix_t * V, int col);

/**
 * Function    : fpica
 * Parameters  : X   -> whitened data as column vectors
 * Return      : A   -> mixing matrix (W is A's inverse)
 * Purpose     : This function runs the deflation ICA method from the Matlab
 *               implementation provided by http://research.ics.aalto.fi/ica/fastica/ .
 *               whitesig must be input as a whitened data matrix with the input
 *               and whitened/mean subtracted/PCA-performed preprocessing steps
 *               already complete. The function will return the mixing matrix A
 *               which contain the ICA components.
 *               We are implementing the deflation approach and using the nonlinearity
 *               function pow3.
 */

// this translation starts at line 582 of fpica.m
matrix_t * fpica (matrix_t * X, matrix_t * dewhiteningMatrix, matrix_t * whiteningMatrix)
{
    int round = 1, numFailures = 0, i = 0;

    int vectorSize = X->cols;
    int numOfIC = vectorSize;

    matrix_t * B = m_zeroes(vectorSize, vectorSize);


    while (round <= numOfIC)
    {
        matrix_t * w = m_zeroes(vectorSize, 1);
        randomize_vector(w);

        matrix_t * transposeB = m_transpose(B);
        matrix_t * tempB = m_product(B, transposeB);
        matrix_t * tempB_prod_w = m_product(tempB, w);

        m_subtract(w, tempB_prod_w);

        m_elem_mult(w, (1/norm(w)));

        wOld = m_zeroes(vectorSize, 1);
        wOld2 = m_zeroes(vectorSize, 1);



        while (i <= maxNumIterations + 1)
        {
            // Project the vector into the space orthogonal to the space
            // spanned by the earlier found basis vectors. Note that we can do
            // the projection with matrix B, since the zero entries do not
            // contribute to the projection.
            transposeB = m_transpose(B);
            tempB = m_product(B, transposeB);
            tempB_prod_w = m_product(tempB, w);

            m_subtract(w, tempB_prod_w);
            m_elem_mult(w, (1/norm(w)));

            copy_w = m_copy(w);
            copy_w2 = m_copy(w);

            m_subtract(copy_w, wOld);
            m_add(copy_w2, wOld);

            if (norm(copy_w) < epsilon || norm(copy_w2) < epsilon)
            {

            }




            m_free(copy_w);
            m_free(copy_w2);

            m_elem_mult(w, (1/norm(w)));
            i++;
        }







        m_free(w);
        m_free(transposeB);
        m_free(tempB);
        m_free(tempB_prod_w);
        m_free(wOld);
        m_free(wOld2);
        round++;
    }


    return NULL;
}

// randomize the vector with values from 0 to 1
void randomize_vector(matrix_t * M)
{
    int i;

    for (i = 0; i < M->rows; i++)
    {
        M->data[i] = ((double)rand() / (double)RAND_MAX);
    }
}


// return the norm of a vector
precision_t norm(matrix_t * M)
{
    int i;
    precision_t sum = 0;

    for (i = 0; i < M->rows; i++)
    {
        sum += pow(M->data[i], 2);
    }

    return sqrt(sum);
}

void m_copy_vector_into_column(matrix_t * M, matrix_t * V, int col)
{
    assert(V->cols == 1);

    memcpy(&elem(M, 0, col), V->data, V->rows * sizeof(precision_t));
}
