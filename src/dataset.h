/**
 * @file dataset.h
 *
 * Interface definitions for the dataset type.
 */
#ifndef DATASET_H
#define DATASET_H

#include <fstream>
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
	std::string path;
	std::vector<data_label_t> labels;
	std::vector<data_entry_t> entries;

	Dataset(const std::string& path);
	Dataset();

	Matrix load_data() const;

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print() const;
};

#endif
