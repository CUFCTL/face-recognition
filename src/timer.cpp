/**
 * @file timer.cpp
 *
 * Implementation of the timer.
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "logger.h"
#include "timer.h"

// Global vector of timer items
std::vector<timer_item_t> items;

/**
 * Start a new timer item.
 *
 * @param name
 */
void timer_push(const char *name)
{
	if ( TIMING ) {
		timer_item_t item;
		item.name = name;
		item.start = clock();
		item.duration = -1;

		items.push_back(item);
	}

	log(LL_VERBOSE, "%s\n", name);
}

/**
 * Stop the most recent timer item which is still running.
 *
 * @param id
 */
void timer_pop(void)
{
	if ( TIMING ) {
		std::vector<timer_item_t>::reverse_iterator iter;

		for ( iter = items.rbegin(); iter != items.rend(); iter++ ) {
			if ( iter->duration == -1 ) {
				break;
			}
		}

		assert(iter != items.rend());

		iter->end = clock();
		iter->duration = (double)(iter->end - iter->start) / CLOCKS_PER_SEC;
	}
}

/**
 * Print all timer items.
 */
void timer_print(void)
{
	if ( TIMING ) {
		std::vector<timer_item_t>::iterator iter;

		// determine the maximum string length
		int max_length = 0;

		for ( iter = items.begin(); iter != items.end(); iter++ ) {
			int len = strlen(iter->name);

			if ( max_length < len ) {
				max_length = len;
			}
		}

		// print timer items
		log(LL_VERBOSE, "\n");
		log(LL_VERBOSE, "Timing Statistics\n");
		log(LL_VERBOSE, "\n");
		log(LL_VERBOSE, "%-*s  %s\n", max_length, "Name", "Duration (s)");
		log(LL_VERBOSE, "%-*s  %s\n", max_length, "----", "------------");

		for ( iter = items.begin(); iter != items.end(); iter++ ) {
			log(LL_VERBOSE, "%-*s  % 12.3lf\n", max_length, iter->name, iter->duration);
		}
		log(LL_VERBOSE, "\n");
	}
}
