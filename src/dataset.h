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
#include "genome.h"
#include "matrix.h"

#define IMAGE_TYPE 1
#define GENOME_TYPE 2

typedef std::string DataLabel;

typedef struct {
	DataLabel label;
	std::string name;
} DataEntry;

class Dataset {
private:
	std::string _path;
	std::vector<DataLabel> _labels;
	std::vector<DataEntry> _entries;

public:
	Dataset(const std::string& path, bool is_labeled=true);
	Dataset();

	inline const std::string& path() const { return this->_path; }
	inline const std::vector<DataLabel>& labels() const { return this->_labels; }
	inline const std::vector<DataEntry>& entries() const { return this->_entries; }

	Matrix load_data(int type) const;

	void save(std::ofstream& file);
	void load(std::ifstream& file);

	void print() const;
};

#endif
