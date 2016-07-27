/**
 * @file image_entry.h
 *
 * Interface definitions for the image entry type.
 *
 * The image entry module is the interface between the database
 * type and the "face database" in the filesystem. As such, this
 * module provides functions to map a directory of images to a
 * collection of image entries as well as various operations on
 * the pathnames of image entries.
 */
#ifndef IMAGE_ENTRY_H
#define IMAGE_ENTRY_H

typedef struct dir {
	char *path;
	int size;
	char **names;
} dir_t;

typedef struct dir_entry {
	int class;
	char *name;
} dir_entry_t;

typedef dir_entry_t image_entry_t;

char * basename(char *path);
int is_same_class(char *path1, char *path2);

int get_directory(const char *path, char ***p_names);
int get_directory_rec(const char *path, dir_entry_t **p_entries, int *p_num_dirs);

#endif
