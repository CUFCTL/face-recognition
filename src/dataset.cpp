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
 * the directory is treated as an observation.
 *
 * If the data are labeled, the filename should be
 * formatted as follows:
 *
 * "<class>_<...>"
 *
 * This format is used to determine the label of each
 * file without separate label data, and to group the
 * entries by label.
 *
 * @param path
 */
Dataset::Dataset(const std::string& path, bool is_labeled)
{
	this->_path = path;

	// get list of files
	struct dirent **files;
	int num_entries = scandir(this->_path.c_str(), &files, is_file, alphasort);

	if ( num_entries <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct entries
	for ( int i = 0; i < num_entries; i++ ) {
		// construct entry name
		std::string name(files[i]->d_name);

		// construct label name
		DataLabel label = is_labeled
			? name.substr(0, name.find_first_of('_'))
			: "";

		// append entry
		this->_entries.push_back(DataEntry {
			label,
			name
		});
	}

	// construct labels
	if ( is_labeled ) {
		for ( const DataEntry& entry : this->_entries ) {
			// search labels for label name
			size_t j = 0;
			while ( j < this->_labels.size() && this->_labels[j] != entry.label ) {
				j++;
			}

			// append label if not found
			if ( j == this->_labels.size() ) {
				this->_labels.push_back(entry.label);
			}
		}
	}

	// clean up
	for ( int i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);
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
 * type is 0 for image, 1 for genome
 */
Matrix Dataset::load_data(int type) const
{
	int i;
	if (type == IMAGE_TYPE)
	{
		// get the image size from the first image
		Image image;

		image.load(this->_path + "/" + this->_entries[0].name);

		// construct image matrix
		int m = image.channels() * image.height() * image.width();
		int n = this->_entries.size();
		Matrix X("X", m, n);

		// map each image to a column in X
		X.image_read(0, image);

		
		for ( i = 1; i < n; i++ ) {
			image.load(this->_path + "/" + this->_entries[i].name);
			X.image_read(i, image);
		}

		return X;
	}
	else if (type == GENOME_TYPE)
	{
		Genome *genome = new Genome();

		genome->load_rna_seq(this->_path + "/" + this->_entries[0].name);

		// construct genome matrix
		int m = genome->gene_count();
		int n = this->_entries.size();
		Matrix X("X", m, n);

		// map each image to a column in X
		X.genome_read(0, genome);

		delete genome;

		for (i = 1; i < n; i++) {
			Genome *genome = new Genome();;
			genome->load_rna_seq(this->_path + "/" + this->_entries[i].name);
			X.genome_read(i, genome);
			delete genome;
		}

		return X;
	}

}

/**
 * Save a dataset to a file.
 *
 * @param file
 */
void Dataset::save(std::ofstream& file)
{
	// save path
	write_string(this->_path.c_str(), file);

	// save labels
	int num_labels = this->_labels.size();
	write_int(num_labels, file);

	for ( const DataLabel& label : this->_labels ) {
		write_string(label.c_str(), file);
	}

	// save entries
	int num_entries = this->_entries.size();
	write_int(num_entries, file);

	for ( const DataEntry& entry : this->_entries ) {
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
	this->_path = read_string(file);

	// read labels
	int num_labels = read_int(file);

	for ( int i = 0; i < num_labels; i++ ) {
		DataLabel label(read_string(file));

		this->_labels.push_back(label);
	}

	// read entries
	int num_entries = read_int(file);

	for ( int i = 0; i < num_entries; i++ ) {
		DataEntry entry;
		entry.label = read_string(file);
		entry.name = read_string(file);

		this->_entries.push_back(entry);
	}
}

/**
 * Print information about a dataset.
 */
void Dataset::print() const
{
	// print path
	log(LL_VERBOSE, "path: %s", this->_path.c_str());
	log(LL_VERBOSE, "");

	// print labels
	log(LL_VERBOSE, "%d classes", this->_labels.size());

	for ( const DataLabel& label : this->_labels ) {
		log(LL_VERBOSE, "%s", label.c_str());
	}
	log(LL_VERBOSE, "");

	// print entries
	log(LL_VERBOSE, "%d entries", this->_entries.size());

	for ( const DataEntry& entry : this->_entries ) {
		log(LL_VERBOSE, "%-8s  %s", entry.label.c_str(), entry.name.c_str());
	}
	log(LL_VERBOSE, "");
}
