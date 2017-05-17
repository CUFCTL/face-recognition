/**
 * @file ica.cpp
 *
 * Implementation of ICA (Hyvarinen, 1999).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ica.h"
#include "logger.h"
#include "math_utils.h"
#include "pca.h"
#include "timer.h"

typedef Matrix (*ica_nonl_func_t)(const Matrix& , const Matrix& );

/**
 * Construct an ICA layer.
 *
 * @param n1
 * @param n2
 * @param nonl
 * @param max_iter
 * @param eps
 */
ICALayer::ICALayer(int n1, int n2, ica_nonl_t nonl, int max_iter, precision_t eps)
{
	this->n1 = n1;
	this->n2 = n2;
	this->nonl = nonl;
	this->max_iter = max_iter;
	this->eps = eps;
}

/**
 * Compute the independent components of a matrix X, which
 * consists of observations in columns.
 *
 * @param X
 * @param y
 * @param c
 */
void ICALayer::compute(const Matrix& X, const std::vector<data_entry_t>& y, int c)
{
	timer_push("ICA");

	timer_push("subtract mean from input matrix");

	// compute mixedsig = X', subtract mean column
	Matrix mixedsig = X.transpose("mixedsig");
	Matrix mixedmean = mixedsig.mean_column("mixedmean");

	mixedsig.subtract_columns(mixedmean);

	timer_pop();

	timer_push("compute whitening matrix and whitened input matrix");

	// compute whitening matrix W_z = inv(sqrt(D)) * W_pca'
	PCALayer pca(this->n1);

	pca.compute(mixedsig, y, c);

	Matrix D_temp1("sqrt(D)", pca.D);
	D_temp1.elem_apply(sqrtf);

	Matrix D_temp2 = D_temp1.inverse("inv(sqrt(D))");
	Matrix W_z = D_temp2.product("W_z", pca.W, false, true);

	// compute whitened input U = W_z * mixedsig
	Matrix U = W_z.product("U", mixedsig);

	timer_pop();

	timer_push("compute mixing matrix");

	// compute mixing matrix
	Matrix W_mix = this->fpica(U, W_z);

	timer_pop();

	timer_push("compute ICA projection matrix");

	// compute independent components
	// icasig = W_mix * (mixedsig + mixedmean * ones(1, mixedsig.cols))
	Matrix ones = Matrix::ones("ones", 1, mixedsig.cols);
	Matrix icasig_temp1 = mixedmean.product("icasig_temp1", ones);

	icasig_temp1.add(mixedsig);

	Matrix icasig = W_mix.product("icasig", icasig_temp1);

	// compute W_ica = icasig'
	this->W = icasig.transpose("W_ica");

	timer_pop();

	timer_pop();
}

/**
 * Project a matrix X into the feature space of an ICA layer.
 *
 * @param X
 */
Matrix ICALayer::project(const Matrix& X)
{
	return this->W.product("P", X, true, false);
}

/**
 * Save an ICA layer to a file.
 *
 * @param file
 */
void ICALayer::save(FILE *file)
{
	this->W.save(file);
}

/**
 * Load an ICA layer from a file.
 *
 * @param file
 */
void ICALayer::load(FILE *file)
{
	this->W.load(file);
}

/**
 * Print information about an ICA layer.
 */
void ICALayer::print()
{
	const char *nonl_name = "";

	if ( this->nonl == ICA_NONL_POW3 ) {
		nonl_name = "pow3";
	}
	else if ( this->nonl == ICA_NONL_TANH ) {
		nonl_name = "tanh";
	}
	else if ( this->nonl == ICA_NONL_GAUSS ) {
		nonl_name = "gauss";
	}

	log(LL_VERBOSE, "ICA\n");
	log(LL_VERBOSE, "  %-20s  %10d\n", "n1", this->n1);
	log(LL_VERBOSE, "  %-20s  %10d\n", "n2", this->n2);
	log(LL_VERBOSE, "  %-20s  %10s\n", "nonl", nonl_name);
	log(LL_VERBOSE, "  %-20s  %10d\n", "max_iter", this->max_iter);
	log(LL_VERBOSE, "  %-20s  %10f\n", "eps", this->eps);
}

/**
 * Compute the parameter update for fpica
 * with the pow3 nonlinearity:
 *
 * g(u) = u^3
 * g'(u) = 3 * u^2
 *
 * which gives:
 *
 * w+ = X * ((X' * w) .^ 3) / X.cols - 3 * w
 *
 * @param w0
 * @param X
 * @return w*
 */
Matrix fpica_pow3 (const Matrix& w0, const Matrix& X)
{
	// compute w+
	Matrix w_temp1 = X.product("w_temp1", w0, true, false);
	w_temp1.elem_apply(pow3);

	Matrix w_temp2("w_temp2", w0);
	w_temp2.elem_mult(3.0f);

	Matrix w = X.product("w", w_temp1);
	w.elem_mult(1.0f / X.cols);
	w.subtract(w_temp2);

	return w;
}

/**
 * Compute the parameter update for fpica
 * with the tanh nonlinearity:
 *
 * g(u) = tanh(u)
 * g'(u) = sech(u)^2 = 1 - tanh(u)^2
 *
 * which gives:
 *
 * w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X.cols
 *
 * @param w0
 * @param X
 * @return w*
 */
Matrix fpica_tanh (const Matrix& w0, const Matrix& X)
{
	// compute w+
	Matrix w_temp1 = X.product("w_temp1", w0, true, false);
	Matrix w_temp2("w_temp2", w_temp1);

	w_temp1.elem_apply(tanhf);
	w_temp2.elem_apply(sechf);
	w_temp2.elem_apply(pow2);

	Matrix w_temp3("w_temp3", w0);
	w_temp3.elem_mult(w_temp2.sum());

	Matrix w = X.product("w", w_temp1);
	w.subtract(w_temp3);
	w.elem_mult(1.0f / X.cols);

	return w;
}

/**
 * Gaussian nonlinearity function.
 *
 * @param x
 */
precision_t gauss (precision_t x)
{
	return x * expf(-(x * x) / 2.0f);
}

/**
 * Derivative of the Gaussian nonlinearity function.
 *
 * @param x
 */
precision_t dgauss (precision_t x)
{
	return (1 - x * x) * expf(-(x * x) / 2.0f);
}

/**
 * Compute the parameter update for fpica
 * with the Gaussian nonlinearity:
 *
 * g(u) = u * exp(-u^2 / 2)
 * g'(u) = (1 - u^2) * exp(-u^2 / 2)
 *
 * which gives:
 *
 * w+ = (X * g(X' * w) - sum(g'(X' * w)) * w) / X.cols
 *
 * @param w0
 * @param X
 * @return w*
 */
Matrix fpica_gauss (const Matrix& w0, const Matrix& X)
{
	// compute w+
	Matrix w_temp1 = X.product("w_temp1", w0, true, false);
	Matrix w_temp2("w_temp2", w_temp1);

	w_temp1.elem_apply(gauss);
	w_temp2.elem_apply(dgauss);

	Matrix w_temp3("w_temp3", w0);
	w_temp3.elem_mult(w_temp2.sum());

	Matrix w = X.product("w", w_temp1);
	w.subtract(w_temp3);
	w.elem_mult(1.0f / X.cols);

	return w;
}

/**
 * Compute the mixing matrix W_mix for an input matrix X using
 * the deflation approach. The input matrix should already
 * be whitened.
 *
 * @param X
 * @param W_z
 * @return mixing matrix W_mix
 */
Matrix ICALayer::fpica(const Matrix& X, const Matrix& W_z)
{
	// if n2 is -1, use default value
	int n2 = (this->n2 == -1)
		? X.rows
		: min(X.rows, this->n2);

	// determine nonlinearity function
	ica_nonl_func_t fpica_update = NULL;

	if ( this->nonl == ICA_NONL_POW3 ) {
		fpica_update = fpica_pow3;
	}
	else if ( this->nonl == ICA_NONL_TANH ) {
		fpica_update = fpica_tanh;
	}
	else if ( this->nonl == ICA_NONL_GAUSS ) {
		fpica_update = fpica_gauss;
	}

	Matrix B = Matrix::zeros("B", n2, n2);
	Matrix W_mix = Matrix::zeros("W_mix", n2, W_z.cols);

	int i;
	for ( i = 0; i < n2; i++ ) {
		log(LL_VERBOSE, "      round %d\n", i + 1);

		// initialize w as a Gaussian (0, 1) random vector
		Matrix w = Matrix::random("w", n2, 1);

		// compute w = (w - B * B' * w), normalize w
		Matrix w_temp1 = B.product("w_temp1", B, false, true);
		Matrix w_temp2 = w_temp1.product("w_temp2", w);

		w.subtract(w_temp2);
		w.elem_mult(1.0f / w.norm());

		// initialize w0
		Matrix w0 = Matrix::zeros("w0", w.rows, w.cols);

		int j;
		for ( j = 0; j < this->max_iter; j++ ) {
			// compute w = (w - B * B' * w), normalize w
			w_temp1 = B.product("w_temp1", B, false, true);
			w_temp2 = w_temp1.product("w_temp2", w);

			w.subtract(w_temp2);
			w.elem_mult(1.0f / w.norm());

			// compute w_delta1 = w - w0
			Matrix w_delta1("w_delta1", w);
			w_delta1.subtract(w0);

			// compute w_delta2 = w + w0
			Matrix w_delta2("w_delta2", w);
			w_delta2.add(w0);

			// determine whether the direction of w and w0 are equal
			precision_t norm1 = w_delta1.norm();
			precision_t norm2 = w_delta2.norm();

			// terminate round if w converges
			if ( norm1 < this->eps || norm2 < this->eps ) {
				// save B(:, i) = w
				B.assign_column(i, w, 0);

				// save W_mix(i, :) = w' * W_z
				Matrix W_temp1 = w.product("W_temp1", W_z, true, false);

				W_mix.assign_row(i, W_temp1, 0);

				// continue to the next round
				break;
			}

			// update w0
			w0.assign_column(0, w, 0);

			// compute w+ based on non-linearity
			w = fpica_update(w0, X);

			// compute w* = w+ / ||w+||
			w.elem_mult(1.0f / w.norm());
		}

		log(LL_VERBOSE, "      iterations: %d\n", j);
	}

	return W_mix;
}
