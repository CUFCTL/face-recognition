/**
 * @file lda.c
 *
 * Implementation of LDA (Belhumeur et al., 1996; Zhao et al., 1998).
 */
#include "database.h"
#include "matrix.h"
#include <stdlib.h>

/**
 * Compute the scatter matrices S_w and S_b for a set of images.
 *
 * @param X        pointer to input matrix
 * @param c        number of classes
 * @param entries  list of entries for each column of X
 * @param S_b      pointer to store between-scatter matrix
 * @param S_w      pointer to store within-scatter matrix
 */
void m_scatter(matrix_t *X, int c, image_entry_t *entries, matrix_t *S_b, matrix_t *S_w)
{
    clock_t scatterTimeBegin = clock();

    matrix_t **X_classes = (matrix_t **)malloc(c * sizeof(matrix_t *));
    matrix_t **U = (matrix_t **)malloc(c * sizeof(matrix_t *));

    // compute the mean of each class
    clock_t meanTimeBegin = clock();
    int i, j;
    for ( i = 0, j = 0; i < c; i++ ) {
        // extract the columns of X in class i
        int k;
        for ( k = j; k < X->cols; k++ ) {
            if ( entries[k].class != entries[j].class ) {
                break;
            }
        }

        X_classes[i] = m_copy_columns(X, j, k);
        j = k;

        // compute the mean of the class
        U[i] = m_mean_column(X_classes[i]);
    }
    clock_t meanTimeEnd = clock();

    // compute the mean of all classes
    clock_t meanAllClassBegin = clock();
    matrix_t *u = m_initialize(X->rows, 1);  // m_mean_column(U);

    for ( i = 0; i < c; i++ ) {
        for ( j = 0; j < X->rows; j++ ) {
            elem(u, j, 0) += elem(U[i], j, 0);
        }
    }
    for ( i = 0; i < X->rows; i++ ) {
        elem(u, i, 0) /= c;
    }
    clock_t meanAllClassEnd = clock();

    // compute the between-scatter S_b = sum(S_b_i, i=1:c)
    // compute the within-scatter S_w = sum(S_w_i, i=1:c)
    double scatterBtwTime = 0;
    double scatterWithinTime = 0;
    for ( i = 0; i < c; i++ ) {
        matrix_t *X_class = X_classes[i];

        // compute S_b_i = n_i * (u_i - u) * (u_i - u)'
        clock_t scatBtwBegin = clock();
        matrix_t *u_i = m_copy(U[i]);
        m_subtract(u_i, u);

        matrix_t *u_i_tr = m_transpose(u_i);
        matrix_t *S_b_i = m_product(u_i, u_i_tr);
        m_elem_mult(S_b_i, X_class->cols);

        m_add(S_b, S_b_i);
        clock_t scatBtwEnd = clock();

        scatterBtwTime += (double)(scatBtwEnd - scatBtwBegin) / CLOCKS_PER_SEC;

        // compute S_w_i = X_class * X_class', X_class is mean-subtracted
        clock_t scatWithinBegin = clock();
        m_subtract_columns(X_class, U[i]);

        matrix_t *X_class_tr = m_transpose(X_class);
        matrix_t *S_w_i = m_product(X_class, X_class_tr);

        m_add(S_w, S_w_i);
        clock_t scatWithinEnd = clock();
        scatterWithinTime += (double)(scatWithinEnd - scatWithinBegin) / CLOCKS_PER_SEC;
        // cleanup
        m_free(u_i);
        m_free(u_i_tr);
        m_free(S_b_i);
        m_free(X_class_tr);
        m_free(S_w_i);
    }

    // cleanup
    for ( i = 0; i < c; i++ ) {
        m_free(X_classes[i]);
        m_free(U[i]);
    }
    free(X_classes);
    free(U);

    clock_t scatterTimeEnd = clock();

    FILE* fp = fopen(FP, "a");
    fprintf(fp, "\nm_scatter, time\n");
    fprintf(fp, "Compute Mean of Each Class, %.3lf\n", (double)(meanTimeEnd - meanTimeBegin) / CLOCKS_PER_SEC);
    fprintf(fp, "Compute Mean of All Classes, %.3lf\n", (double)(meanAllClassEnd - meanAllClassBegin) / CLOCKS_PER_SEC);
    fprintf(fp, "Compute between-scatter S_b, %.3lf\n", scatterBtwTime);
    fprintf(fp, "Compute within-scatter S_w, %.3lf\n", scatterWithinTime);
    fprintf(fp, "Total, %.3lf\n", (double)(scatterTimeEnd - scatterTimeBegin) / CLOCKS_PER_SEC);
    fclose(fp);
}

/**
 * Compute the projection matrix of a training set with LDA.
 *
 * @param W_pca_tr  PCA projection matrix
 * @param X         image matrix
 * @param c         number of classes
 * @param entries   list of entries for each image
 * @param n_opt1    number of columns in W_pca to use
 * @param n_opt2    number of columns in W_fld to use
 * @return projection matrix W_lda'
 */
matrix_t * LDA(matrix_t *W_pca_tr, matrix_t *X, int c, image_entry_t *entries, int n_opt1, int n_opt2)
{
	  clock_t timeLDABegin = clock();

    // use only the first n_opt1 columns in W_pca
    clock_t scatterTimeBegin = clock();
    matrix_t *W_pca = m_transpose(W_pca_tr);
    matrix_t *W_pca2 = m_copy_columns(W_pca, 0, n_opt1);
    matrix_t *W_pca2_tr = m_transpose(W_pca2);
    matrix_t *P_pca = m_product(W_pca2_tr, X);

    // compute scatter matrices S_b and S_w
    matrix_t *S_b = m_zeros(P_pca->rows, P_pca->rows);
    matrix_t *S_w = m_zeros(P_pca->rows, P_pca->rows);

    m_scatter(P_pca, c, entries, S_b, S_w);
    clock_t scatterTimeEnd = clock();

    // compute W_fld = eigenvectors of S_w^-1 * S_b
	  clock_t eigenTimeBegin = clock();
    matrix_t *S_w_inv = m_inverse(S_w);
    matrix_t *J = m_product(S_w_inv, S_b);
    matrix_t *J_eval = m_initialize(J->rows, 1);
    matrix_t *J_evec = m_initialize(J->rows, J->cols);

    m_eigen(J, J_eval, J_evec);
	  clock_t eigenTimeEnd = clock();

    // take only the first n_opt2 columns in J_evec
    clock_t wLDATimeBegin = clock();
    matrix_t *W_fld = m_copy_columns(J_evec, 0, n_opt2);

    matrix_t *W_fld_tr = m_transpose(W_fld);

    // compute W_lda' = W_fld' * W_pca'
    matrix_t *W_lda_tr = m_product(W_fld_tr, W_pca2_tr);
    clock_t wLDATimeEnd = clock();

    m_free(W_pca);
    m_free(W_pca2);
    m_free(W_pca2_tr);
    m_free(P_pca);
    m_free(S_b);
    m_free(S_w);
    m_free(S_w_inv);
    m_free(J);
    m_free(J_eval);
    m_free(J_evec);
    m_free(W_fld);
    m_free(W_fld_tr);

	clock_t timeLDAEnd = clock();

	FILE* fp = fopen(FP, "a");
  fprintf(fp, "\nLDA, Time\n");
  fprintf(fp, "Compute Scatter Matrices, %.3lf\n", (double)(scatterTimeEnd - scatterTimeBegin) / CLOCKS_PER_SEC);
	fprintf(fp, "Compute Eigen Vectors, %.3lf\n", (double)(eigenTimeEnd - eigenTimeBegin) / CLOCKS_PER_SEC);
  fprintf(fp, "Compute W_lda_tr, %.3lf\n", (double)(wLDATimeEnd - wLDATimeBegin) / CLOCKS_PER_SEC);
  fprintf(fp, "Overall time for LDA, %.3lf\n", (double)(timeLDAEnd - timeLDABegin) / CLOCKS_PER_SEC);
  fclose(fp);

	//printf("Time taken to compute Scatter Matrix: %.2lf \n", (double)(scatterTimeEnd - scatterTimeBegin) / CLOCKS_PER_SEC);
	//printf("Time taken to compute Eigen Vectors: %.2lf \n", (double)(eigenTimeEnd - eigenTimeBegin) / CLOCKS_PER_SEC);
	//printf("Time taken to coomplete LDA: %.2lf \n", (double)(timeEnd - timeBegin) / CLOCKS_PER_SEC);

    return W_lda_tr;
}
