/**
 * @file logger.cpp
 *
 * Implementation of the logger.
 */
#include <stdarg.h>
#include <stdio.h>
#include "logger.h"

logger_level_t LOGLEVEL = LL_INFO;

/**
 * Log a message with a given loglevel.
 *
 * This function uses the same argument format
 * as printf().
 *
 * @param level
 * @param format
 */
void log(logger_level_t level, const char *format, ...)
{
	if ( level <= LOGLEVEL ) {
		FILE *stream = (level <= LL_ERROR)
			? stderr
			: stdout;
		va_list ap;

		va_start(ap, format);
		vfprintf(stream, format, ap);
		va_end(ap);
	}
}
