/**
 * @file dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include "matrix.h"

typedef struct {
	int id;
	char *name;
} data_label_t;

typedef struct {
	data_label_t *label;
	char *name;
} data_entry_t;

typedef struct {
	int num_labels;
	data_label_t *labels;

	int num_entries;
	data_entry_t *entries;
} dataset_t;

dataset_t * dataset_construct(const char *path);
void dataset_destruct(dataset_t *dataset);

void dataset_fwrite(dataset_t *dataset, FILE *file);
dataset_t * dataset_fread(FILE *file);

matrix_t * dataset_load(dataset_t *dataset);

void dataset_print_labels(dataset_t *dataset);
void dataset_print_entries(dataset_t *dataset);

#endif
