/**
 * @file timing.cpp
 *
 * Functions for timing code
 */
#include "timing.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>

// Global vector of timing values
std::vector<timing_info_t> v;

/**
 * Start a new timing item.
 *
 * @param name
 */
void timing_push(const char *name)
{
if ( TIMING ) {
	timing_info_t ti;
	ti.name = name;
	ti.start = clock();
	ti.duration = -1;

	v.push_back(ti);
}
}

/**
 * Stop the most recent timing item which is still running.
 *
 * @param id
 */
void timing_pop(void)
{
if ( TIMING ) {
	std::vector<timing_info_t>::reverse_iterator iter;

	for ( iter = v.rbegin(); iter != v.rend(); iter++ ) {
		if ( iter->duration == -1 ) {
			break;
		}
	}

	assert(iter != v.rend());

	iter->end = clock();
	iter->duration = (double)(iter->end - iter->start) / CLOCKS_PER_SEC;
}
}

/**
 * Print all timing items.
 */
void timing_print(void)
{
if ( TIMING ) {
	std::vector<timing_info_t>::iterator iter;

	// determine the maximum string length
	int max_length = 0;

	for ( iter = v.begin(); iter != v.end(); iter++ ) {
		int len = strlen(iter->name);

		if ( max_length < len ) {
			max_length = len;
		}
	}

	// print timing items
	putchar('\n');
	printf("Timing Statistics\n");
	putchar('\n');
	printf("%-*s  %s\n", max_length, "Name", "Duration (s)");
	printf("%-*s  %s\n", max_length, "----", "------------");

	for ( iter = v.begin(); iter != v.end(); iter++ ) {
		printf("%-*s  % 12.3lf\n", max_length, iter->name, iter->duration);
	}
	putchar('\n');
}
}
