/**
 * @file dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <string>
#include <vector>
#include "matrix.h"

typedef std::string data_label_t;

typedef struct {
	data_label_t label;
	std::string name;
} data_entry_t;

class Dataset {
public:
	std::vector<data_label_t> labels;
	std::vector<data_entry_t> entries;

	Dataset(const std::string& path);
	Dataset(FILE *file);
	Dataset();

	void save(FILE *file);

	Matrix load() const;

	void print_labels() const;
	void print_entries() const;
};

#endif
