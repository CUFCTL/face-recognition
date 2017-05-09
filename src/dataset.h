/**
 * @file dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <vector>
#include "matrix.h"

typedef struct {
	int id;
	char *name;
} data_label_t;

typedef struct {
	char *label;
	char *name;
} data_entry_t;

class Dataset {
public:
	std::vector<data_label_t> labels;
	std::vector<data_entry_t> entries;

	Dataset(const char *path);
	Dataset(FILE *file);
	Dataset();

	void save(FILE *file);

	matrix_t *load() const;

	void print_labels() const;
	void print_entries() const;
};

#endif
