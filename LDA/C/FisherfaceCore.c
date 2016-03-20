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

TODO : update debug prints to more elegant solution
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
    int verbose = 1;

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
    L = m_matrix_multiply(A, At, A->cols);
    m_free(At);

    if (verbose) {
        printf("\nL = surrogate of covariance:\n");
        matrix_print(L, 2);
    }

    // Calculate eigenvectors and eigenvalues

    // D is the (D)iagonal matrix of eigenvalues
    // V is the matrix of eigen(V)ectors
    m_eigenvalues_eigenvectors(M, D, V);

    if (verbose) {
        printf("D, eigenvalues:\n");
        matrix_print(D, 2);
        printf("V, eigenvectors:\n");
        matrix_print(V, 4);
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
        matrix_print(L_eig_vec, 4);
    }

    //**************************************************************************
    //Calculating the eigenvectors of covariance matrix 'C'

    V_PCA = m_matrix_multiply(A, L_eig_vec, A->cols);

    if (verbose) {
        printf("V_PCA:\n");
        matrix_print(V_PCA, 4);
    }

    //**************************************************************************
    //Projecting centered image vectors onto eigenspace
    //<.m: 55-61>

    ProjectedImages_PCA = matrix_constructor(P - Class_number, P);

    for (i = 0; i < P; i++) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, P - Class_number, 1, pixels, 1,     *V_PCA->data, V_PCA->cols, &A->data[0][i], A->cols, 0, &ProjectedImages_PCA->data[0][i], ProjectedImages_PCA->cols);
    }

    if (verbose) {
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
    
    if (verbose) {
        printf("m_PCA:\n");
        matrix_print(m_PCA, 16);
    }
    
    J_eig_vec = matrix_construct(P-Class_number,P-Class_number);
    alphai = matrix_construct(P-Class_number,1);
    alphar = matrix_construct(P-Class_number,1);
    beta = matrix_construct(P-Class_number,1);
    
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
