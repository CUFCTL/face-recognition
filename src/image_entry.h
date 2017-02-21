/**
 * @file image_entry.h
 *
 * Interface definitions for the image entry type.
 *
 * The image entry module is the interface between the database
 * and the filesystem.
 */
#ifndef IMAGE_ENTRY_H
#define IMAGE_ENTRY_H

typedef struct {
	int id;
	char *name;
} image_label_t;

typedef struct {
	image_label_t *label;
	char *name;
} image_entry_t;

int get_directory(const char *path, image_entry_t **p_entries, int *p_num_labels, image_label_t **p_labels);

void debug_print_labels(image_label_t *labels, int num);
void debug_print_entries(image_entry_t *entries, int num);

#endif
