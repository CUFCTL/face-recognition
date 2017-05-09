/**
 * @file bayes.h
 *
 * Interface definitions for the naive Bayes classifier.
 */
#ifndef BAYES_H
#define BAYES_H

#include "dataset.h"
#include "matrix.h"

char ** bayes(matrix_t *X, std::vector<data_entry_t>& Y, std::vector<data_label_t>& C, matrix_t *X_test);

#endif
