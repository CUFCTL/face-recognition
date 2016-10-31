/**
 * @file pca.c
 *
 * Implementation of PCA (Turk and Pentland, 1991).
 */
#include "database.h"
#include "matrix.h"

/**
 * Compute the principal components of a training set.
 *
 * Currently, this function returns all of the n computed
 * eigenvectors, where n is the number of training images.
 *
 * @param X  mean-subtracted image matrix
 * @return projection matrix W_pca'
 */
matrix_t * PCA(matrix_t *X, matrix_t **L_eval, matrix_t **W_pca)
{
	clock_t PCAbegin = clock();
	// compute the surrogate matrix L = X' * X
	clock_t computeLBegin = clock();
	matrix_t *X_tr = m_transpose(X);
	matrix_t *L = m_product(X_tr, X);

	m_free(X_tr);
	clock_t computeLEnd = clock();

	// compute eigenvectors for L
	clock_t eigenLBegin = clock();
	*L_eval = m_initialize(L->rows, 1);
	matrix_t *L_evec = m_initialize(L->rows, L->cols);

	m_eigen(L, *L_eval, L_evec);
	clock_t eigenLEnd = clock();

	// compute eigenfaces W_pca = X * L_evec
	clock_t eigenFacesBegin = clock();
	*W_pca = m_product(X, L_evec);
	matrix_t *W_pca_tr = m_transpose(*W_pca);

	m_free(L);
	clock_t eigenFacesEnd = clock();

	clock_t PCAend = clock();

	FILE* fp = fopen(FP, "a");
	fprintf(fp, "\nPCA, Time\n");
	fprintf(fp, "Compute Surrogate Matrix L, %.3lf\n", (double)(computeLEnd - computeLBegin) / CLOCKS_PER_SEC);
	fprintf(fp, "Compute Eigen Vectors for L, %.3lf\n", (double)(eigenLEnd - eigenLBegin) / CLOCKS_PER_SEC);
	fprintf(fp, "Compute Eigen Faces for L, %.3lf\n", (double)(eigenFacesEnd - eigenFacesBegin) / CLOCKS_PER_SEC);
	fprintf(fp, "Total, %.3lf\n", (double)(PCAend - PCAbegin) / CLOCKS_PER_SEC);
	fclose(fp);
	return W_pca_tr;
}
