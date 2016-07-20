/**
 * @file image_entry.c
 *
 * Implementation of the image entry type.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_entry.h"

/**
 * Get whether an entry is a PGM or PPM image based
 * on the file extension.
 *
 * @param entry
 * @return 1 if entry is PGM or PPM image, 0 otherwise
 */
int is_valid_image(const struct dirent *entry)
{
	return strstr(entry->d_name, ".pgm") != NULL
		|| strstr(entry->d_name, ".ppm") != NULL;
}

/**
 * Get whether an entry is a directory, excluding "." and "..".
 *
 * @param entry
 * @return 1 if entry is a directory, 0 otherwise
 */
int is_directory(const struct dirent *entry)
{
	return entry->d_type == DT_DIR
		&& strcmp(entry->d_name, ".") != 0
		&& strcmp(entry->d_name, "..") != 0;
}

/**
 * Get a list of images in a directory.
 *
 * @param path
 * @param image_names
 * @return number of images that were found
 */
int get_image_names(const char *path, char ***image_names)
{
	// get list of image entries
	struct dirent **entries;
	int num_images = scandir(path, &entries, is_valid_image, alphasort);

	if ( num_images <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct list of image paths
	*image_names = (char **)malloc(num_images * sizeof(char *));

	int i;
	for ( i = 0; i < num_images; i++ ) {
		(*image_names)[i] = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf((*image_names)[i], "%s/%s", path, entries[i]->d_name);
	}

	// clean up
	for ( i = 0; i < num_images; i++ ) {
		free(entries[i]);
	}
	free(entries);

	return num_images;
}

/**
 * Get a list of image entries in a directory.
 *
 * The directory should contain a subdirectory for each
 * class, and each subdirectory should contain the images
 * of its class.
 *
 * @param path
 * @param image_entries
 * @param num_classes
 * @return number of images that were found
 */
int get_image_entries(const char *path, image_entry_t **image_entries, int *num_classes)
{
	// get list of classes
	struct dirent **entries;
	*num_classes = scandir(path, &entries, is_directory, alphasort);

	if ( (*num_classes) <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// get list of image names for each class
	char ***classes = (char ***)malloc((*num_classes) * sizeof(char **));
	int *class_sizes = (int *)malloc((*num_classes) * sizeof(int *));

	int num_images = 0;

	int i;
	for ( i = 0; i < (*num_classes); i++ ) {
		char *class_path = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf(class_path, "%s/%s", path, entries[i]->d_name);

		class_sizes[i] = get_image_names(class_path, &classes[i]);
		num_images += class_sizes[i];

		free(class_path);
	}

	// flatten image name lists into a list of entries
	*image_entries = (image_entry_t *)malloc(num_images * sizeof(image_entry_t));

	int num, j;
	for ( num = 0, i = 0; i < (*num_classes); i++ ) {
		for ( j = 0; j < class_sizes[i]; j++ ) {
			(*image_entries)[num] = (image_entry_t) {
				.class = i,
				.name = classes[i][j]
			};

			num++;
		}
	}

	// clean up
	for ( i = 0; i < (*num_classes); i++ ) {
		free(entries[i]);
		free(classes[i]);
	}
	free(entries);
	free(classes);
	free(class_sizes);

	return num_images;
}
