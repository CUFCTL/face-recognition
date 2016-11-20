/**
 * @file timing.h
 *
 * Data structure for timing code
 */
#ifndef TIMING_H
#define TIMING_H

#include <time.h>

extern int TIMING;

typedef struct {
	const char *name;
	clock_t start;
	clock_t end;
	double duration;
} timing_info_t;

void timing_push(const char *);
void timing_pop(void);
void timing_print(void);

#endif
