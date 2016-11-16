/**
 * @file timing.cpp
 *
 * Functions for timing code
 */
#include "timing.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

// Global vector of timing values
std::vector<timing_info_t> v;

/**
 * Add new timing unit to vector, start timer
 *
 */
void timing_start(const char *name)
{
if (TIMING) {
	timing_info_t ti;

	ti.name = name;
	ti.start = clock();

	v.push_back(ti);
}
}

/**
 * Stop timer
 */
void timing_end(const char *name)
{
if (TIMING) {
	std::vector<timing_info_t>::iterator n;

	for (n = v.begin(); n != v.end(); n++) {
		if (strcmp(n->name, name) == 0) break;
	}

	// If not true at runtime, bug exists in usage
	assert(strcmp(n->name, name) == 0);

	n->end = clock();
	n->duration = (double) (n->end - n->start) / CLOCKS_PER_SEC;
}
}

/**
 * Print all timing values
 */
void timing_print(void)
{
if (TIMING) {
	std::vector<timing_info_t>::iterator n;
	n = v.begin();

	printf("\n\n");
	printf("Timing Statistics\n");
	printf("Name, Duration\n");

	do {
		printf("%s, ", n->name);
		printf("%.3lf\n", n->duration);
		n++;
	} while (n != v.end());

	printf("\n\n");
}
}
