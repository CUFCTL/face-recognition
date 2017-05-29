/**
 * @file dataset.cpp
 *
 * Implementation of the dataset type.
 */
#include <dirent.h>
#include "dataset.h"
#include "logger.h"

/**
 * Get whether an entry is a file, excluding "." and "..".
 *
 * @param entry
 */
int is_file(const struct dirent *entry)
{
	std::string name(entry->d_name);

	return (name != "." && name != "..");
}

/**
 * Read an integer from a binary file.
 *
 * @param file
 */
int read_int(std::ifstream& file)
{
	int n;
	file.read(reinterpret_cast<char *>(&n), sizeof(int));

	return n;
}

/**
 * Read a string from a binary file.
 *
 * @param file
 */
std::string read_string(std::ifstream& file)
{
	int num = read_int(file);

	char *buffer = new char[num];
	file.read(buffer, num);

	std::string str(buffer);

	delete[] buffer;

	return str;
}

/**
 * Write an integer to a binary file.
 *
 * @param file
 */
void write_int(int n, std::ofstream& file)
{
	file.write(reinterpret_cast<char *>(&n), sizeof(int));
}

/**
 * Write a string to a file.
 *
 * @param str
 * @param file
 */
void write_string(const std::string& str, std::ofstream& file)
{
	int num = str.size() + 1;

	write_int(num, file);
	file.write(str.c_str(), num);
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
		entry.name = filename;

		entries.push_back(entry);
	}

	// clean up
	for ( i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);

	// construct dataset
	this->path = path;
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
 * Load the data matrix X for a dataset. Each column
 * in X is an observation. Every observation in X must
 * have the same dimensionality.
 *
 * This function assumes that the data are images.
 */
Matrix Dataset::load_data() const
{
	// get the image size from the first image
	Image image;

	image.load(this->path + "/" + this->entries[0].name);

	// construct image matrix
	int m = image.channels * image.height * image.width;
	int n = this->entries.size();
	Matrix X("X", m, n);

	// map each image to a column in X
	X.image_read(0, image);

	int i;
	for ( i = 1; i < n; i++ ) {
		image.load(this->path + "/" + this->entries[i].name);
		X.image_read(i, image);
	}

	return X;
}

/**
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(std::ofstream& file)
{
	// save path
	write_string(this->path.c_str(), file);

	// save labels
	int num_labels = this->labels.size();
	write_int(num_labels, file);

	for ( const data_label_t& label : this->labels ) {
		write_string(label.c_str(), file);
	}

	// save entries
	int num_entries = this->entries.size();
	write_int(num_entries, file);

	for ( const data_entry_t& entry : this->entries ) {
		write_string(entry.label.c_str(), file);
		write_string(entry.name.c_str(), file);
	}
}

/**
 * Load a dataset from a file.
 *
 * @param file
 */
void Dataset::load(std::ifstream& file)
{
	// read path
	this->path = read_string(file);

	// read labels
	int num_labels = read_int(file);

	for ( int i = 0; i < num_labels; i++ ) {
		data_label_t label(read_string(file));

		this->labels.push_back(label);
	}

	// read entries
	int num_entries = read_int(file);

	for ( int i = 0; i < num_entries; i++ ) {
		data_entry_t entry;
		entry.label = read_string(file);
		entry.name = read_string(file);

		this->entries.push_back(entry);
	}
}

/**
 * Print information about a dataset.
 */
void Dataset::print() const
{
	// print path
	log(LL_VERBOSE, "path: %s", this->path.c_str());
	log(LL_VERBOSE, "");

	// print labels
	log(LL_VERBOSE, "%d classes", this->labels.size());

	for ( const data_label_t& label : this->labels ) {
		log(LL_VERBOSE, "%s", label.c_str());
	}
	log(LL_VERBOSE, "");

	// print entries
	log(LL_VERBOSE, "%d entries", this->entries.size());

	for ( const data_entry_t& entry : this->entries ) {
		log(LL_VERBOSE, "%8s  %s", entry.label.c_str(), entry.name.c_str());
	}
	log(LL_VERBOSE, "");
}
