/**
 * @file image_entry.h
 *
 * Interface definitions for the image entry type.
 */
#ifndef IMAGE_ENTRY_H
#define IMAGE_ENTRY_H

typedef struct {
	int class;
	char *name;
} image_entry_t;

int get_image_names(const char *path, char ***image_names);
int get_image_entries(const char *path, image_entry_t **image_entries, int *num_classes);

#endif
