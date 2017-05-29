/**
 * @file timer.h
 *
 * Interface definitions for the timer.
 */
#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <string>
#include <vector>

typedef struct {
	std::string name;
	int level;
	clock_t start;
	clock_t end;
	float duration;
} timer_item_t;

typedef struct {
	std::vector<timer_item_t> items;
	int level;
} timekeeper_t;

void timer_push(const std::string& name);
float timer_pop(void);
void timer_print(void);

#endif
