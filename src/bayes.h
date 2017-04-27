/**
 * @file bayes.h
 *
 * Interface definitions for the naive Bayes classifier.
 */
#ifndef BAYES_H
#define BAYES_H

#include "dataset.h"
#include "matrix.h"

data_label_t ** bayes(matrix_t *X, data_entry_t *Y, data_label_t *C, int num_classes, matrix_t *X_test);

#endif
