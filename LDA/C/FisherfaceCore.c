/*******************************************************************************
 Use Principle Component Analysis (PCA) and Fisher Linear Discriminant (FLD) to determine the most
 discriminating features between images of faces.

 Description: This function gets a 2D matrix, containing all training image vectors
 and returns 4 outputs which are extracted from training database.
 Suppose Ti is a training image, which has been reshaped into a 1D vector.
 Also, P is the total number of MxN training images and C is the number of
 classes. At first, centered Ti is mapped onto a (P-C) linear subspace by V_PCA
 transfer matrix: Zi = V_PCA * (Ti - m_database).
 Then, Zi is converted to Yi by projecting onto a (C-1) linear subspace, so that
 images of the same class (or person) move closer together and images of difference
 classes move further apart: Yi = V_Fisher' * Zi = V_Fisher' * V_PCA' * (Ti - m_database)

 Argument:
    D                             - ((M*N)xP) A 2D matrix, containing all 1D image vectors.
                                     All of 1D column vectors have the same length of M*N,
                                     and 'D' will be a MNxP 2D matrix.

 Returns:
    M                             - matrix_t ** consisting of the following 4 entries:
    M[0] = mean                   - ((M*N)x1) Mean of the training database
    M[1] = V_PCA                  - ((M*N)x(P-C)) Eigen vectors of the covariance matrix of the training database
    M[2] = V_Fisher               - ((P-C)x(C-1)) Largest (C-1) eigen vectors of matrix J = inv(Sw) * Sb
    M[3] = ProjectedImages_Fisher - ((C-1)xP) Training images, which are projected onto Fisher linear space

 See also: EIG

 Original version by Amir Hossein Omidvarnia, October 2007
                     Email: aomidvar@ece.ut.ac.ir
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../../MatrixOperations/matrixOperations.c"
#include "CreateDatabase.h"
#include "FisherfaceCore.h"

matrix_t **FisherfaceCore(const matrix_t *Database)
{
    int Class_population = IM_PER_PERSON; //Set value according to database (Images per person)
    int P = Database->images; //Total Number of training images
    int pixels = Database->pixels; //total pixels per image (i.e., width * height)
    int Class_number = P / Class_population; //Number of classes (or persons)
    int i, j, k, l;
    double *work, *info;    // Array of doubles containing intermediate values for dggev
    double temp_double;

    // Verbose mode: debug prints
    int verbose = 0;

    // matrix_t types
    matrix_t **M; //What the function returns
    matrix_t *Database_matrix; //Database stored in matrix_t form
    matrix_t *m_database; //Pixelwise mean of database images
    matrix_t *A; //Deviation matrix (imagewise difference from mean)
    matrix_t *At; // Transpose of A
    matrix_t *L; //Surrogate of covariance matrix, L = A' * A
    matrix_t *D; //Eigenvalues of L
    matrix_t *V; //Eigenvectors of L
    matrix_t *L_eig_vec; //filtered eigenvectors
    matrix_t *V_PCA; //
    matrix_t *ProjectedImages_PCA;
    matrix_t *m_PCA; //mean of ProjectedImages_PCA
    matrix_t *tempMean; //temporary matrix to store the mean of ProjectedImages_PCA
    matrix_t *tempMat;
    matrix_t *S;
    matrix_t *J_eig_vec;
    matrix_t *alphai, *alphar, *beta;

    // Allocate room for four return matrices
    M = (matrix_t **) malloc(4 * sizeof(matrix_t *));

    if (verbose) {
        printf("Database\n");
        m_fprint(stdout, Database_matrix);
    }

    //**************************************************************************
    //Calculate mean
    m_database = m_meanCols(Database_matrix);

    // This value is the first returned by the function
    M[0] = m_database;

    if (verbose) {
        printf("\nmean:\n");
        m_fprint(stdout, M[0]);
    }

    //**************************************************************************
    //Calculate A, deviation matrix
    A = matrix_constructor(pixels, P);

    for (i = 0; i < pixels; i++) {
        // each column in A->data is the difference between an image and the mean
        for (j = 0; j < P; j++) {
            A->data[i][j] = Database->data[i][j] - m_database->data[i][0];
        }
    }

    if (verbose) {
        printf("\ndeviation:\n");
        m_fprintf(stdout, A);
    }

    //**************************************************************************
    //Calculate L, surrogate of covariance matrix, L = A'*A

    At = m_transpose(A);
    L = m_product(A, At, A->cols);
    m_free(At);

    if (verbose) {
        printf("\nL = surrogate of covariance:\n");
        m_fprintf(stdout, L);
    }

    // Calculate eigenvectors and eigenvalues

    // D is the (D)iagonal matrix of eigenvalues
    // V is the matrix of eigen(V)ectors
    m_eigenvalues_eigenvectors(M, D, V);

    if (verbose) {
        printf("D, eigenvalues:\n");
        m_fprintf(stdout, D);
        printf("V, eigenvectors:\n");
        m_fprintf(stdout, V);
    }

    //**************************************************************************
    //Sorting and eliminating small eigenvalues

    L_eig_vec = matrix_constructor(P, P - Class_number);

    for (i = 0; i < L_eig_vec->rows; i++) {
        for (j = 0; j < L_eig_vec->cols; j++) {
            L_eig_vec->data[i][j] = V->data[i][j];
        }
    }

    if (verbose) {
        printf("L_eig_vec, trimmed eigenvectors:\n");
        m_fprintf(stdout, L_eig_vec);
    }

    //**************************************************************************
    //Calculating the eigenvectors of covariance matrix 'C'

    V_PCA = m_product(A, L_eig_vec, A->cols);

    if (verbose) {
        printf("V_PCA:\n");
        m_fprintf(stdout, V_PCA);
    }

    //**************************************************************************
    //Projecting centered image vectors onto eigenspace
    // Each column in ProjectedImages_PCA is an image, transposed and multiplied
    // by the eigenvectors of the difference database to produce the column

    ProjectedImages_PCA = m_product(V_PCA, A, A->cols);

    if (verbose) {
        printf("ProjectedImages_PCA:\n");
        m_fprintf(stdout, ProjectedImages_PCA);
    }

    //**************************************************************************
    //Calculating the mean of each class in eigenspace

    m_PCA = matrix_mean(ProjectedImages_PCA);

    if (verbose) {
        printf("m_PCA:\n");
        m_fprintf(stdout, m_PCA);
    }

    // Scatter Matrices
    m_scatter(m_PCA, Sw, Sb);

    if (verbose) {
        printf("Scatter Matrices:\n");
        m_fprintf(stdout, Sw);
        m_fprintf(stdout, Sb);
    }

    // Fisher bases
    // Eigenvalues of scatter matrices
    m_gen_eigenvalues(Sb, Sw, J_eig_vec, J_eig_val);

    if (verbose) {
        printf("Fisher Space:\n");
        m_fprintf(stdout, J_eig_vec);
    }

    // Copy largest eigenvectors
    for (i = 0; i < Class_number - 1; i++) {
        for (j = 0; j < P; j++) {
            V_Fisher[i][j] = J_eig_vec[i][j];
        }
    }

    // Project images onto Fisher-Space
    ProjectedImages_Fisher = m_project(V_Fisher, ProjectedImages_PCA);

    if (verbose) {
        printf("Fisher Projected Images\n");
        m_fprintf(stdout, ProjectedImages_Fisher);
    }

    //**************************************************************************

	//FREE INTERMEDIATES
    matrix_destructor(Database_matrix);
    matrix_destructor(A);
    matrix_destructor(V);
    matrix_destructor(D);
    matrix_destructor(L_eig_vec);
    matrix_destructor(V_PCA);
    matrix_destructor(ProjectedImages_PCA);

    return M;
}

// scatter
void m_scatter(matrix_t *mat, matrix_t *Sw, matrix_t *Sb)
{
    // TODO put in mat lib
    for (i = 1; i <= Class_number; i++)
    {
        tempMean = matrix_bounded_mean(ProjectedImages_PCA, 0, ProjectedImages_PCA->rows-1, (i-1)*Class_population+1, i*Class_population);

        for (j = 0; j < m->rows; j++)
        {
           m->data[j][i] = tempMean->data[j][1];
        }

        matrix_destructor(tempMean);

        for (j = ((i-1)*Class_population+1); j <= (i*Class_population); j++)
        {
            tempMat = matrix_constructor(P-Class_number, 1);

            for (k = 0; k < (P-Class_number); k++)
            {
                tempMat->data[k][0] = ProjectedImages_PCA->data[k][j] - m->data[k][i];
            }

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tempMat->rows, 1, 1, 1,
                        *tempMat->data, tempMat->cols, *tempMat->data, tempMat->cols, 1, *S->data, S->cols);

        }

        for (j = 0; j < Sw->rows; j++)
        {
           for (k = 0; k < Sw->rows; k++)
           {
               Sw->data[j][k] += S->data[j][k];
           }
        }

        for (k = 0; k < Sb->rows; k++)
        {
            tempMat->data[k][0] = m[k][i] - m_PCA[k][0];
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tempMat->rows, 1, 1, 1,
                        *tempMat->data, tempMat->cols, *tempMat->data, tempMat->cols, 1, *Sb->data, Sb->cols);
    }
}
