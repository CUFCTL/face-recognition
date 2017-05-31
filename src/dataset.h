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
private:
	std::string _path;
	std::vector<data_label_t> _labels;
	std::vector<data_entry_t> _entries;

public:
	Dataset(const std::string& path);
	Dataset();

	inline const std::string& path() const { return this->_path; }
	inline const std::vector<data_label_t>& labels() const { return this->_labels; }
	inline const std::vector<data_entry_t>& entries() const { return this->_entries; }

	Matrix load_data() const;

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print() const;
};

#endif
