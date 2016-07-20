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
 * Get a pathname with the first directory prefix removed.
 *
 * @param path
 * @return pointer to index in path
 */
char * basename(char *path)
{
	char *s = strchr(path, '/');

	return s != NULL
		? s + 1
		: NULL;
}

/**
 * Get whether an entry is a file, excluding "." and "..".
 *
 * @param entry
 * @return 1 if entry is a file, 0 otherwise
 */
int is_file(const struct dirent *entry)
{
	return strcmp(entry->d_name, ".") != 0
		&& strcmp(entry->d_name, "..") != 0;
}

/**
 * Get a list of files in a directory.
 *
 * @param path
 * @param p_names
 * @return number of files that were found
 */
int get_directory(const char *path, char ***p_names)
{
	// get list of entries
	struct dirent **entries;
	int num_entries = scandir(path, &entries, is_file, alphasort);

	if ( num_entries <= 0 ) {
		perror("scandir");
		exit(1);
	}

	// construct list of file names
	char **names = (char **)malloc(num_entries * sizeof(char *));

	int i;
	for ( i = 0; i < num_entries; i++ ) {
		names[i] = (char *)malloc(strlen(path) + 1 + strlen(entries[i]->d_name) + 1);

		sprintf(names[i], "%s/%s", path, entries[i]->d_name);
	}

	// clean up
	for ( i = 0; i < num_entries; i++ ) {
		free(entries[i]);
	}
	free(entries);

	*p_names = names;

	return num_entries;
}

/**
 * Get a list of files in a two-level directory tree.
 *
 * @param path
 * @param p_entries
 * @param p_num_dirs
 * @return number of files that were found
 */
int get_directory_rec(const char *path, dir_entry_t **p_entries, int *p_num_dirs)
{
	// get list of directories
	char **paths;
	int num_dirs = get_directory(path, &paths);

	// get list of names in each directory
	dir_t *dirs = (dir_t *)malloc(num_dirs * sizeof(dir_t));

	int i;
	for ( i = 0; i < num_dirs; i++ ) {
		dirs[i].path = paths[i];
		dirs[i].size = get_directory(dirs[i].path, &dirs[i].names);
	}

	// count the total number of entries
	int num_entries = 0;

	for ( i = 0; i < num_dirs; i++ ) {
		num_entries += dirs[i].size;
	}

	// flatten directory lists into one list of entries
	dir_entry_t *entries = (dir_entry_t *)malloc(num_entries * sizeof(dir_entry_t));

	int num = 0;
	int j;
	for ( i = 0; i < num_dirs; i++ ) {
		for ( j = 0; j < dirs[i].size; j++ ) {
			entries[num] = (dir_entry_t) {
				.class = i,
				.name = dirs[i].names[j]
			};

			num++;
		}
	}

	// clean up
	for ( i = 0; i < num_dirs; i++ ) {
		free(paths[i]);
	}
	free(paths);
	free(dirs);

	*p_entries = entries;
	*p_num_dirs = num_dirs;

	return num_entries;
}
