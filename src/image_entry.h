/**
 * @file image_entry.h
 *
 * Interface definitions for the image entry type.
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
