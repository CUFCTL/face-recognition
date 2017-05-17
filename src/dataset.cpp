/**
 * @file dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

/**
 * Get whether an entry is a file, excluding "." and "..".
 *
 * @param entry
 * @return 1 if entry is a file, 0 otherwise
 */
int is_file(const struct dirent *entry)
{
	return (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0);
}

/**
 * Read a string from a file.
 *
 * @param file
 * @return pointer to new string
 */
char * str_fread(FILE *file)
{
	int num;
	fread(&num, sizeof(int), 1, file);

	char *str = (char *)malloc(num * sizeof(char));
	fread(str, sizeof(char), num, file);

	return str;
}

/**
 * Write a string to a file.
 *
 * @param str
 * @param file
 */
void str_fwrite(const char *str, FILE *file)
{
	int num = strlen(str) + 1;
	fwrite(&num, sizeof(int), 1, file);
	fwrite(str, sizeof(char), num, file);
}

/**
 * Construct a dataset from a directory. Each file in
 * the directory is treated as an observation. The
 * filename should be formatted as follows:
 *
 * "<class>_<...>"
 *
 * This format is used to determine the label of each
 * file without separate label data, and to order the
 * entries by class.
 *
 * @param path
 */
Dataset::Dataset(const std::string& path)
{
	// get list of files
	struct dirent **files;
	int num_entries = scandir(path.c_str(), &files, is_file, alphasort);

	if ( num_entries <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct labels and entries
	std::vector<data_label_t> labels;
	std::vector<data_entry_t> entries;

	int i;
	for ( i = 0; i < num_entries; i++ ) {
		// get filename
		std::string filename(files[i]->d_name);

		// construct label name
		unsigned n = filename.find_first_of('_');
		data_label_t label = filename.substr(0, n);

		// search labels for label name
		unsigned j = 0;
		while ( j < labels.size() && labels[j] != label ) {
			j++;
		}

		// append label if not found
		if ( j == labels.size() ) {
			labels.push_back(label);
		}

		// append entry
		data_entry_t entry;
		entry.label = labels[j];
		entry.name = path + "/" + filename;

		entries.push_back(entry);
	}

	// clean up
	for ( i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);

	// construct dataset
	this->labels = labels;
	this->entries = entries;
}

/**
 * Construct a dataset from a file.
 *
 * @param file
 */
Dataset::Dataset(FILE *file)
{
	// read labels
	int num_labels;
	std::vector<data_label_t> labels;

	fread(&num_labels, sizeof(int), 1, file);

	int i;
	for ( i = 0; i < num_labels; i++ ) {
		data_label_t label(str_fread(file));

		labels.push_back(label);
	}

	// read entries
	int num_entries;
	std::vector<data_entry_t> entries;

	fread(&num_entries, sizeof(int), 1, file);

	for ( i = 0; i < num_entries; i++ ) {
		data_entry_t entry;
		entry.label = str_fread(file);
		entry.name = str_fread(file);

		entries.push_back(entry);
	}

	this->labels = labels;
	this->entries = entries;
}

/**
 * Construct an empty dataset.
 */
Dataset::Dataset()
{
}

/**
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(FILE *file)
{
	// save labels
	int num_labels = this->labels.size();
	fwrite(&num_labels, sizeof(int), 1, file);

	int i;
	for ( i = 0; i < num_labels; i++ ) {
		str_fwrite(this->labels[i].c_str(), file);
	}

	// save entries
	int num_entries = this->entries.size();
	fwrite(&num_entries, sizeof(int), 1, file);

	for ( i = 0; i < num_entries; i++ ) {
		str_fwrite(this->entries[i].label.c_str(), file);
		str_fwrite(this->entries[i].name.c_str(), file);
	}
}

/**
 * Get the data matrix X for a dataset. Each column
 * in X is an observation. Every observation in X must
 * have the same dimensionality.
 *
 * This function assumes that the data are images.
 *
 * @return pointer to data matrix
 */
Matrix Dataset::load() const
{
	// get the image size from the first image
	Image image;

	image.load(this->entries[0].name.c_str());

	// construct image matrix
	int m = image.channels * image.height * image.width;
	int n = this->entries.size();
	Matrix X("X", m, n);

	// map each image to a column in X
	X.image_read(0, image);

	int i;
	for ( i = 1; i < n; i++ ) {
		image.load(this->entries[i].name.c_str());
		X.image_read(i, image);
	}

	return X;
}

/**
 * Print the labels in a dataset.
 */
void Dataset::print_labels() const
{
	unsigned i;
	for ( i = 0; i < this->labels.size(); i++ ) {
		printf("%3d  %s\n", i, this->labels[i].c_str());
	}
	putchar('\n');
}

/**
 * Print the entries in a dataset.
 */
void Dataset::print_entries() const
{
	unsigned i;
	for ( i = 0; i < this->entries.size(); i++ ) {
		printf("%8s  %s\n", this->entries[i].label.c_str(), this->entries[i].name.c_str());
	}
	putchar('\n');
}
