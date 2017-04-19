/**
 * @file dataset.cpp
 *
 * Implementation of the image entry type.
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
 * Construct a dataset from a directory. Each file in
 * the directory is treated as an observation. The
 * filename should be formatted as follows:
 *
 * "<class>_<...>"
 *
 * This format is used to determine the label of each
 * file without separate label data.
 *
 * @param path
 * @return pointer to dataset
 */
dataset_t *dataset_construct(const char *path)
{
	// get list of files
	struct dirent **files;
	int num_entries = scandir(path, &files, is_file, alphasort);

	if ( num_entries <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct lists of entries, labels
	data_entry_t *entries = (data_entry_t *)malloc(num_entries * sizeof(data_entry_t));
	data_label_t *labels = (data_label_t *)malloc(num_entries * sizeof(data_label_t));
	int num_labels = 0;

	int i;
	for ( i = 0; i < num_entries; i++ ) {
		// set entry name
		char *filename = files[i]->d_name;

		entries[i].name = (char *)malloc(strlen(path) + 1 + strlen(filename) + 1);
		sprintf(entries[i].name, "%s/%s", path, filename);

		// construct label name
		int n = strchr(filename, '_') - filename;
		char *label_name = strndup(filename, n);

		// search labels for label name
		int j = 0;
		while ( j < num_labels && strcmp(labels[j].name, label_name) != 0 ) {
			j++;
		}

		// append label if not found
		if ( j == num_labels ) {
			labels[j].id = num_labels;
			labels[j].name = label_name;
			num_labels++;
		}
		else {
			free(label_name);
		}

		// set entry label
		entries[i].label = &labels[j];
	}

	// truncate labels
	labels = (data_label_t *)realloc(labels, num_labels * sizeof(data_label_t));

	// clean up
	for ( i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);

	// construct dataset
	dataset_t *dataset = (dataset_t *)malloc(sizeof(dataset_t));

	dataset->num_entries = num_entries;
	dataset->entries = entries;

	dataset->num_labels = num_labels;
	dataset->labels = labels;

	return dataset;
}

/**
 * Destruct a dataset.
 *
 * @param dataset
 */
void dataset_destruct(dataset_t *dataset)
{
	// free entries
	int i;
	for ( i = 0; i < dataset->num_entries; i++ ) {
		free(dataset->entries[i].name);
	}
	free(dataset->entries);

	// free labels
	for ( i = 0; i < dataset->num_labels; i++ ) {
		free(dataset->labels[i].name);
	}
	free(dataset->labels);
}

/**
 * Save a dataset to a file.
 *
 * @param dataset
 * @param file
 */
void dataset_fwrite(dataset_t *dataset, FILE *file)
{
	// save labels
	fwrite(&dataset->num_labels, sizeof(int), 1, file);

	int i;
	for ( i = 0; i < dataset->num_labels; i++ ) {
		fwrite(&dataset->labels[i].id, sizeof(int), 1, file);

		int num = strlen(dataset->labels[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(dataset->labels[i].name, sizeof(char), num, file);
	}

	// save entries
	fwrite(&dataset->num_entries, sizeof(int), 1, file);

	for ( i = 0; i < dataset->num_entries; i++ ) {
		fwrite(&dataset->entries[i].label->id, sizeof(int), 1, file);

		int num = strlen(dataset->entries[i].name) + 1;
		fwrite(&num, sizeof(int), 1, file);
		fwrite(dataset->entries[i].name, sizeof(char), num, file);
	}
}

/**
 * Load a dataset from a file.
 *
 * @param file
 * @return pointer to new dataset
 */
dataset_t * dataset_fread(FILE *file)
{
	dataset_t *dataset = (dataset_t *)malloc(sizeof(dataset_t));

	// read labels
	fread(&dataset->num_labels, sizeof(int), 1, file);

	dataset->labels = (data_label_t *)malloc(dataset->num_labels * sizeof(data_label_t));

	int i;
	for ( i = 0; i < dataset->num_labels; i++ ) {
		fread(&dataset->labels[i].id, sizeof(int), 1, file);

		int num;
		fread(&num, sizeof(int), 1, file);

		dataset->labels[i].name = (char *)malloc(num * sizeof(char));
		fread(dataset->labels[i].name, sizeof(char), num, file);
	}

	// read entries
	fread(&dataset->num_entries, sizeof(int), 1, file);

	dataset->entries = (data_entry_t *)malloc(dataset->num_entries * sizeof(data_entry_t));

	for ( i = 0; i < dataset->num_entries; i++ ) {
		int label_id;
		fread(&label_id, sizeof(int), 1, file);

		dataset->entries[i].label = &dataset->labels[label_id];

		int num;
		fread(&num, sizeof(int), 1, file);

		dataset->entries[i].name = (char *)malloc(num * sizeof(char));
		fread(dataset->entries[i].name, sizeof(char), num, file);
	}

	return dataset;
}

/**
 * Get the data matrix X for a dataset. Each column
 * in X is an observation. Every observation in X must
 * have the same dimensionality.
 *
 * This function assumes that the data are images.
 *
 * @param dataset
 * @return pointer to data matrix
 */
matrix_t * dataset_load(dataset_t *dataset)
{
	// get the image size from the first image
	image_t *image = image_construct();
	image_read(image, dataset->entries[0].name);

	// construct image matrix
	int m = image->channels * image->height * image->width;
	int n = dataset->num_entries;
	matrix_t *X = m_initialize("X", m, n);

	// map each image to a column in X
	m_image_read(X, 0, image);

	int i;
	for ( i = 1; i < n; i++ ) {
		image_read(image, dataset->entries[i].name);
		m_image_read(X, i, image);
	}

	image_destruct(image);

	return X;
}

/**
 * Print the labels in a dataset.
 *
 * @param dataset
 */
void dataset_print_labels(dataset_t *dataset)
{
	int i;
	for ( i = 0; i < dataset->num_labels; i++ ) {
		printf("%3d  %s\n", dataset->labels[i].id, dataset->labels[i].name);
	}
	putchar('\n');
}

/**
 * Print the entries in a dataset.
 *
 * @param dataset
 */
void dataset_print_entries(dataset_t *dataset)
{
	int i;
	for ( i = 0; i < dataset->num_entries; i++ ) {
		printf("%8s  %s\n", dataset->entries[i].label->name, dataset->entries[i].name);
	}
	putchar('\n');
}
