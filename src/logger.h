/**
 * @file logger.h
 *
 * Interface definitions for the logger.
 */
#ifndef LOGGER_H
#define LOGGER_H

typedef enum logger_level_t {
	LL_ERROR   = -1,
	LL_WARN    =  0,
	LL_INFO    =  1,
	LL_VERBOSE =  2,
	LL_DEBUG   =  3
} loggger_level_t;

extern logger_level_t LOGLEVEL;

#define LOGGER(level) ((level) <= LOGLEVEL)

void log(logger_level_t level, const char *format, ...);

#endif
