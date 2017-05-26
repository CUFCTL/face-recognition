/**
 * @file matrix.h
 *
 * Interface definitions for the matrix type.
 *
 * NOTE: Unlike C, which stores static arrays in row-major
 * order, this library stores matrices in column-major order.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include "image.h"

typedef float precision_t;

#define M_ELEM_FPRINT  "% 10.4g"

#define ELEM(M, i, j) (M).data[(j) * (M).rows + (i)]

typedef precision_t (*elem_func_t)(precision_t);

class Matrix {
public:
	const char *name;
	int rows;
	int cols;
	precision_t *data;
	precision_t *data_gpu;

	// constructor, destructor functions
	Matrix(const char *name, int rows, int cols);
	Matrix(const char *name, int rows, int cols, precision_t *data);
	Matrix(const char *name, const Matrix& M);
	Matrix(const char *name, const Matrix& M, int i, int j);
	Matrix(const Matrix& M);
	Matrix();
	~Matrix();

	static Matrix identity(const char *name, int rows);
	static Matrix ones(const char *name, int rows, int cols);
	static Matrix random(const char *name, int rows, int cols);
	static Matrix zeros(const char *name, int rows, int cols);

	// I/O functions
	void print(FILE *file) const;
	void save(FILE *file) const;
	void load(FILE *file);

	void gpu_read();
	void gpu_write();

	void image_read(int i, const Image& image);
	void image_write(int i, Image& image);

	// getter functions
	int argmax() const;
	Matrix diagonalize(const char *name) const;
	void eigen(const char *V_name, const char *D_name, int n1, Matrix& V, Matrix& D) const;
	Matrix inverse(const char *name) const;
	Matrix mean_column(const char *name) const;
	Matrix mean_row(const char *name) const;
	precision_t norm() const;
	Matrix product(const char *name, const Matrix& B, bool transA=false, bool transB=false) const;
	precision_t sum() const;
	Matrix transpose(const char *name) const;

	// mutator functions
	void add(const Matrix& B);
	void assign_column(int i, const Matrix& B, int j);
	void assign_row(int i, const Matrix& B, int j);
	void elem_apply(elem_func_t f);
	void elem_mult(precision_t c);
	void subtract(const Matrix& B);
	void subtract_columns(const Matrix& a);
	void subtract_rows(const Matrix& a);

	// operators
	inline Matrix& operator=(Matrix B) { swap(*this, B); return *this; };
	inline Matrix& operator+=(const Matrix& B) { this->add(B); return *this; };
	inline Matrix& operator-=(const Matrix& B) { this->subtract(B); return *this; };
	inline Matrix& operator*=(precision_t c) { this->elem_mult(c); return *this; };
	inline Matrix& operator/=(precision_t c) { this->elem_mult(1 / c); return *this; };

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};

inline Matrix operator+(Matrix A, const Matrix& B) { A += B; return A; }
inline Matrix operator-(Matrix A, const Matrix& B) { A -= B; return A; }
inline Matrix operator*(const Matrix& A, const Matrix& B) { return A.product("", B); }
inline Matrix operator*(Matrix A, precision_t c) { A *= c; return A; }
inline Matrix operator/(Matrix A, precision_t c) { A /= c; return A; }

#endif
