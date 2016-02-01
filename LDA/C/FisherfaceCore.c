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
    M                             - MATRIX ** consisting of the following 4 entries:
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
#include <cblas.h>
#include <lapacke.h>
#include <assert.h>

#include "ppm.h"
#include "CreateDatabase.h"
#include "matrix.h"
#include "FisherfaceCore.h"

MATRIX **FisherfaceCore(const database_t *Database)
{
    int Class_population = 4; //Set value according to database (Images per person)
    int P = Database->images; //Total Number of training images
    int pixels = Database->pixels; //total pixels per image (i.e., width * height)
    int Class_number = P / Class_population; //Number of classes (or persons)
    int i, j, k, l;
    // debug print flags
    int p_database = 0;
    int p_mean = 0;
    int p_dev = 0;
    int p_cov = 0;
    int p_eig = 1;
    int p_vpca = 0;
    int p_pipca = 1;
    int p_mPCA = 1;
    double *work, *info;    // Array of doubles containing intermediate values for dggev
    double temp_double;

    // MATRIX types
    MATRIX **M; //What the function returns
    MATRIX *Database_matrix; //Database stored in MATRIX form
    MATRIX *m_database; //Pixelwise mean of database images
    MATRIX *A; //Deviation matrix (imagewise difference from mean)
    MATRIX *L; //Surrogate of covariance matrix, L = A' * A
    MATRIX *D; //Eigenvalues of L
    MATRIX *V; //Eigenvectors of L
    MATRIX *L_eig_vec; //filtered eigenvectors
    MATRIX *V_PCA; //
    MATRIX *ProjectedImages_PCA;
    MATRIX *m_PCA; //mean of ProjectedImages_PCA
    MATRIX *tempMean; //temporary matrix to store the mean of ProjectedImages_PCA
    MATRIX *tempMat;
    MATRIX *S;
    MATRIX *J_eig_vec;
    MATRIX *alphai, *alphar, *beta;

    M = (MATRIX **) malloc(4 * sizeof(MATRIX *));

    // Convert Database to MATRIX
    Database_matrix = matrix_constructor(pixels, Database->images);
    for (i = 0; i < Database_matrix->rows; i++) {
        for (j = 0; j < Database_matrix->cols; j++) {
            Database_matrix->data[i][j] = Database->data[i][j];
        }
    }

    if (p_database) {
        printf("Database\n");
        matrix_print(Database_matrix, 0);
    }

    //**************************************************************************
    //Calculate mean
    //<.m: 36>
    m_database = matrix_mean(Database_matrix);

    //Assign mean database
    M[0] = m_database;

    if (p_mean) {
        printf("\nmean:\n");
        matrix_print(M[0], 2);
    }

    //**************************************************************************
    //Calculate A, deviation matrix
    //<.m: 39>
    A = matrix_constructor(pixels, P);

    for (i = 0; i < pixels; i++) {
        // each column in A->data is the difference between an image and the mean
        for (j = 0; j < P; j++) {
            A->data[i][j] = Database->data[i][j] - m_database->data[i][0];
        }
    }

    if (p_dev) {
        printf("\ndeviation:\n");
        matrix_print(A, 2);
    }

    //**************************************************************************
    //Calculate L, surrogate of covariance matrix, L = A'*A
    //<.m: 42>
    L = matrix_constructor(P, P);

  //cblas_dgemm(Order,         TransA,     TransB,       M,       N,       K,       alpha, A,        lda,     B,        ldb,     beta, C,        ldc);
//  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, P,       P,       pixels,  1,     *A->data, P,       *A->data, P,       0,    *L->data, P);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A->cols, A->cols, A->rows, 1,     *A->data, A->cols, *A->data, A->cols, 0,    *L->data, L->cols);

    if (p_cov) {
        printf("\nL = surrogate of covariance:\n");
        matrix_print(L, 2);
    }

    // Calculate eigenvectors and eigenvalues
    //<.m: 43>
    D = matrix_constructor(P, 1);

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', P, *L->data, P, *D->data);
    V = L;
    L = NULL;

    if (p_eig) {
        printf("D, eigenvalues:\n");
        matrix_print(D, 2);
        printf("V, eigenvectors:\n");
        matrix_print(V, 4);
    }

    //**************************************************************************
    //Sorting and eliminating small eigenvalues
    //<.m: 46-50>

    L_eig_vec = matrix_constructor(P, P - Class_number);

    for (i = 0; i < L_eig_vec->rows; i++) {
        for (j = 0; j < L_eig_vec->cols; j++) {
            L_eig_vec->data[i][j] = V->data[i][j];
        }
    }

    if (p_eig) {
        printf("L_eig_vec, trimmed eigenvectors:\n");
        matrix_print(L_eig_vec, 4);
    }

    //**************************************************************************
    //Calculating the eigenvectors of covariance matrix 'C'
    //<.m: 54>

    V_PCA = matrix_constructor(pixels, P - Class_number);

    //void cblas_dgemm(Order,         TransA,       TransB,       M,      N,                K, alpha, *A,       lda,     *B,               ldb,             beta, *C,           ldc);
    cblas_dgemm(       CblasRowMajor, CblasNoTrans, CblasNoTrans, pixels, P - Class_number, P, 1,     *A->data, A->cols, *L_eig_vec->data, L_eig_vec->cols, 0,    *V_PCA->data, V_PCA->cols);

    if (p_vpca) {
        printf("V_PCA:\n");
        matrix_print(V_PCA, 4);
    }

    //**************************************************************************
    //Projecting centered image vectors onto eigenspace
    //<.m: 55-61>

    ProjectedImages_PCA = matrix_constructor(P - Class_number, P);

    for (i = 0; i < P; i++) {
        //cblas_dgemm(Order,       TransA,     TransB,       M,                N, K,      alpha, A,            lda,         B,          ldb,     beta, C,                          ldc);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, P - Class_number, 1, pixels, 1,     *V_PCA->data, V_PCA->cols, &A->data[0][i], A->cols, 0, &ProjectedImages_PCA->data[0][i], ProjectedImages_PCA->cols);
    }

    if (p_pipca) {
        printf("ProjectedImages_PCA:\n");
        matrix_print(ProjectedImages_PCA, 2);
    }

    //**************************************************************************
    //Calculating the mean of each class in eigenspace
    //<.m: 64>
//    m_PCA = mean(ProjectedImages_PCA,2); % Total mean in eigenspace
//    m = zeros(P-Class_number,Class_number);
//    Sw = zeros(P-Class_number,P-Class_number); % Initialization of Within Scatter Matrix
//    Sb = zeros(P-Class_number,P-Class_number); % Initialization of Between Scatter Matrix
//
//    for i = 1 : Class_number
//        m(:,i) = mean( ( ProjectedImages_PCA(:,((i-1)*Class_population+1):i*Class_population) ), 2 )';
//
//        S  = zeros(P-Class_number,P-Class_number);
//        for j = ( (i-1)*Class_population+1 ) : ( i*Class_population )
//            S = S + (ProjectedImages_PCA(:,j)-m(:,i))*(ProjectedImages_PCA(:,j)-m(:,i))';
//        end
//
//        Sw = Sw + S; % Within Scatter Matrix
//        Sb = Sb + (m(:,i)-m_PCA) * (m(:,i)-m_PCA)'; % Between Scatter Matrix
//    end

    m_PCA = matrix_mean(ProjectedImages_PCA);

    m = matrix_constructor(P-Class_number,Class_number);
    Sw = matrix_constructor(P-Class_number, P-Class_number);
    Sb = matrix_constructor(P-Class_number, P-Class_number);
    S = matrix_constructor(P-Class_number,P-Class_number);

    
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
    
    if (p_mPCA) {
        printf("m_PCA:\n");
        matrix_print(m_PCA, 16);
    }
    
    J_eig_vec = matrix_construct(P-Class_number,P-Class_number);
    alphai = matrix_construct(P-Class_number,1);
    alphar = matrix_construct(P-Class_number,1);
    beta = matrix_construct(P-Class_number,1);
    
    // 'N' - do not compute left eig. values
    // 'V' - compute right eig. values
    // Sb->cols - the size of the matrices (square)
    // Sb->data - the matrix A in eig(A,B)
    // Sb->rows - the leading dimension of A
    // Sw->data - the matrix B in eig(A,B)
    // Sw->rows - the leading dimension of B
    // alphai, alphar, beta - dummy values
    // J_eig_vec - matrix to output eigenvectors
    // work - integer array output containing intermediate values
    // -1 - automatically determine the size of work
    // info - array containing informataion about dggev performance
    
    ddgev('N', 'V', Sb->cols, Sb->data, Sb->rows, Sw->data, Sw->rows, alphar, alphai, beta, NULL, 1, J_eig_vec, J_eig_vec->rows, work, -1, info);

    for (i = 0; i < P-ClassNumber; i++)
    {
        for (j = 0; j < P-ClassNumber / 2; j++)
        {
            temp_double = J_eig_vec->data[i][j];
            J_eig_vec->data[i][j] = J_eig_vec->data[i][P-ClassNumber-j];
            J_eig_vec->data[i][P-ClassNumber-j] = temp_double;
        }
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

void DestroyFisher(MATRIX **M)
{
    matrix_destructor(M[0]);
//    matrix_destructor(M[1]);
//    matrix_destructor(M[2]);
//    matrix_destructor(M[3]);
    free(M);
}
