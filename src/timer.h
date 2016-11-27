/**
 * @file timer.h
 *
 * Interface definitions for the timer.
 */
#ifndef TIMER_H
#define TIMER_H

#include <time.h>

extern int TIMING;

typedef struct {
	const char *name;
	clock_t start;
	clock_t end;
	double duration;
} timer_item_t;

void timer_push(const char *);
void timer_pop(void);
void timer_print(void);

#endif
