/**
 * @file logger.cpp
 *
 * Implementation of the logger.
 */
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
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

		time_t t = time(NULL);
		struct tm *tm = localtime(&t);

		fprintf(stream, "[%04d-%02d-%02d %02d:%02d:%02d] ",
			1900 + tm->tm_year, 1 + tm->tm_mon, tm->tm_mday,
			tm->tm_hour, tm->tm_min, tm->tm_sec);

		va_start(ap, format);
		vfprintf(stream, format, ap);
		va_end(ap);
	}
}
