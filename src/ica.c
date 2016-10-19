/**
 * @file ica.c
 *
 * Implementation of ICA (Bartlett et al., 2002; Draper et al., 2003).
 */
 #include "database.h"
 #include "matrix.h"
 #include <assert.h>
 #include <math.h>
 #include <stdlib.h>
 #include <string.h>


#define maxNumIterations 1000
#define epsilon 0.0001

matrix_t * diagonalize(matrix_t *M);
void m_elem_sqrt (matrix_t * M);
void randomize_vector(matrix_t * m);
precision_t norm(matrix_t * m);
void m_copy_vector_into_column(matrix_t * M, matrix_t * V, int col);
matrix_t * fpica (matrix_t * X, matrix_t * dewhiteningMatrix, matrix_t * whiteningMatrix);

// THIS FILE SHOULD PREPROCESS THE DATA THAT WILL BE CALLED USING THE FPICA ALGORITHM
//  * WHITEN THE DATA
//  * CALCULATE PCA

/**
 * Compute the whitening matrix W_z for a matrix X.
 *
 * The whitening matrix, when applied to X, removes
 * the first- and second-order statistics; that is,
 * the mean and covariances are set to zero and the
 * variances are equalized.
 *
 * @param X  mean-subtracted input matrix
 * @return whitening matrix W_z
 */
matrix_t * sphere (matrix_t *X, matrix_t *E, matrix_t *D, matrix_t **whiteningMatrix, matrix_t **dewhiteningMatrix)
{
    // compute whitened data based on whitenv.m
    m_elem_sqrt(D);
    matrix_t *inv_sqrt_D = m_inverse(D);
    matrix_t *E_tr = m_transpose(E);
    *whiteningMatrix = m_product(inv_sqrt_D, E_tr);
    *dewhiteningMatrix = m_product(E, D);

    matrix_t * newVectors = m_product(*whiteningMatrix, X);

    // cleanup
    m_free(inv_sqrt_D);
    m_free(E_tr);

    return newVectors;
}

// L_eval.... eigenvalue vector... must translate this to diagonal matrix
// L_evec.... eigenvector matrix
matrix_t * ICA (matrix_t *X, matrix_t *L_eval, matrix_t *L_evec)
{
    // call spherex after diagonalizing the eigenvalues
    matrix_t * whiteningMatrix, * dewhiteningMatrix;
    matrix_t * D = diagonalize(L_eval); // DEBUG: L_eval is 360x1 for orl_faces
    matrix_t * whitesig = sphere(X, L_evec, D, &whiteningMatrix, &dewhiteningMatrix);

    // call fpica(whitened_matrix)
    // A is the mixing matrix
    matrix_t *A = fpica(whitesig, dewhiteningMatrix, whiteningMatrix);

    // cleanup
    m_free(D);
    m_free(whiteningMatrix);
    m_free(dewhiteningMatrix);
    m_free(whitesig);

    return A;
}

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
    int round = 1, i = 0;

    int numSamples = X->cols; // make sure this and vectorSize are correct
    int vectorSize = X->rows;
    int numOfIC = vectorSize;

    matrix_t * B = m_zeros(vectorSize, vectorSize);
    matrix_t * A = m_zeros(vectorSize, vectorSize);
    matrix_t * W = m_zeros(vectorSize, vectorSize);

    while (round <= numOfIC)
    {
        matrix_t * w = m_zeros(vectorSize, 1);
        randomize_vector(w);

        matrix_t * transposeB = m_transpose(B);
        matrix_t * tempB = m_product(B, transposeB);
        matrix_t * tempB_prod_w = m_product(tempB, w);

        m_subtract(w, tempB_prod_w);

        m_elem_mult(w, (1/norm(w)));

        matrix_t * wOld = m_zeros(vectorSize, 1);
        matrix_t * wOld2 = m_zeros(vectorSize, 1);


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

            matrix_t * copy_w = m_copy(w);
            matrix_t * copy_w2 = m_copy(w);

            m_subtract(copy_w, wOld);
            m_add(copy_w2, wOld);

            // line 680 of fpica.m
            if (norm(copy_w) < epsilon || norm(copy_w2) < epsilon)
            {
                int c;

                // save the vector w to the matrix B
                for (c = 0; c < vectorSize; c++)
                {
                    elem(B, c, round) = elem(w, c, 0);
                }

                // calculate the de-whitened vector
                matrix_t * temp_A_vec = m_product(dewhiteningMatrix, w);

                // save the de-whitened vector to matrix A
                for (c = 0; c < vectorSize; c++)
                {
                    elem(A, c, round) = elem(temp_A_vec, c, 0);
                }

                m_free(temp_A_vec);

                // calculate ICA filter
                matrix_t * temp_W_vec = m_product(m_transpose(w), whiteningMatrix);

                // save the ICA filter vector
                for (c = 0; c < vectorSize; c++)
                {
                    elem(W, round, c) = elem(temp_W_vec, 0, c);
                }

            }

            m_free(temp_W_vec);

            wOld2 = wOld;
            wOld = w;

            // pow3 function on w on line 767 of fpica.m
            matrix_t * X_tran_w = m_product(m_transpose(X), w);

            // raise each element to the third power
            for (c = 0; c < vectorSize; c++)
            {
                elem(X_tran_w, c, 0) = pow(elem(X_tran_w), 3);
            }

            matrix_t * w_cpy = m_copy(w);
            m_elem_mult(w_cpy, 3);
            w = m_product(X, X_tran_w);
            m_elem_mult(w, (1/numSamples));
            m_subtract(w, w_cpy);

            m_free(w_cpy);
            m_free(X_tran_w);

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

matrix_t * diagonalize (matrix_t *M)
{
    int i;

    matrix_t * D = m_zeros(M->rows, M->rows);

    for (i = 0; i < M->rows; i++)
    {
        elem(D, i, i) = elem(M, i, 0);
    }

    return D;
}

void m_elem_sqrt (matrix_t * M)
{
    int i, j;

    for (i = 0; i < M->rows; i++)
    {
        for (j = 0; j < M->cols; j++)
        {
            elem(M, i , j) = sqrt(elem(M, i, j));
        }
    }
}
