/**
 * @file timing.h
 *
 * Data structure for timing code
 */
#ifndef TIMING_H
#define TIMING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>

extern int TIMING;

/**
 * Timing information
 *
 * Name of functional unit
 * Start time
 * Finish time
 */
typedef struct {
	const char *name;
	clock_t start;
	clock_t end;
	double duration;
} timing_info_t;

void timing_start(const char *);
void timing_end(const char *);
void timing_print(void);

#ifdef __cplusplus
}
#endif

#endif
