/**
 * @file image_entry.cpp
 *
 * Implementation of the image entry type.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_entry.h"

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
 * Map a directory of files to a collection of image entries with labels.
 *
 * @param path
 * @param p_entries
 * @param p_num_labels
 * @param p_labels
 * @return number of image entries
 */
int get_directory(const char *path, image_entry_t **p_entries, int *p_num_labels, image_label_t **p_labels)
{
	// get list of files
	struct dirent **files;
	int num_entries = scandir(path, &files, is_file, alphasort);

	if ( num_entries <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct lists of entries, labels
	image_entry_t *entries = (image_entry_t *)malloc(num_entries * sizeof(image_entry_t));
	image_label_t *labels = (image_label_t *)malloc(num_entries * sizeof(image_label_t));
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
	labels = (image_label_t *)realloc(labels, num_labels * sizeof(image_label_t));

	// clean up
	for ( i = 0; i < num_entries; i++ ) {
		free(files[i]);
	}
	free(files);

	// save outputs
	*p_entries = entries;
	*p_num_labels = num_labels;
	*p_labels = labels;

	return num_entries;
}

/**
 * Print a list of labels.
 *
 * @param labels
 * @param num
 */
void debug_print_labels(image_label_t *labels, int num)
{
	int i;
	for ( i = 0; i < num; i++) {
		printf("%3d  %s\n", labels[i].id, labels[i].name);
	}
	putchar('\n');
}

/**
 * Print a list of entries.
 *
 * @param entries
 * @param num
 */
void debug_print_entries(image_entry_t *entries, int num)
{
	int i;
	for ( i = 0; i < num; i++) {
		printf("%8s  %s\n", entries[i].label->name, entries[i].name);
	}
	putchar('\n');
}
