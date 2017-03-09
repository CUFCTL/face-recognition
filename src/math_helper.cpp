/**
 * @file math_helper.cpp
 *
 * Library of helpful math functions.
 */
#include <math.h>
#include "math_helper.h"

typedef struct {
	void *id;
	int count;
} item_count_t;

/**
 * Determine the max of two integers.
 *
 * @param x
 * @param y
 */
int max(int x, int y)
{
	return x > y ? x : y;
}

/**
 * Determine the min of two integers.
 *
 * @param x
 * @param y
 */
int min(int x, int y)
{
	return x < y ? x : y;
}

/**
 * Determine the most frequently occuring item in an
 * array of items based on an identification function.
 *
 * @param base
 * @param nmemb
 * @param size
 * @param identify
 */
void * mode(const void *base, size_t nmemb, size_t size, void * (*identify)(const void *))
{
	item_count_t *counts = (item_count_t *)calloc(nmemb, sizeof(item_count_t));

	// compute the frequency of each item in the list
	size_t i;
	for ( i = 0; i < nmemb; i++ ) {
		void *item = (char *)base + i * size;
		void *id = identify(item);

		int n = 0;
		while ( counts[n].id != NULL && counts[n].id != id ) {
			n++;
		}

		counts[n].id = id;
		counts[n].count++;
	}

	// find the item with the highest frequency
	void *max_id = NULL;
	int max_count = 0;

	for ( i = 0; i < nmemb && counts[i].id != NULL; i++ ) {
		if ( max_id == NULL || max_count < counts[i].count ) {
			max_id = counts[i].id;
			max_count = counts[i].count;
		}
	}

	free(counts);

	return max_id;
}

/**
 * Generate a normally-distributed (mu, sigma) random number
 * using the Box-Muller transform.
 *
 * @param mu      mean
 * @param sigma   standard deviation
 * @return normally-distributed random number
 */
float rand_normal(float mu, float sigma)
{
	static int init = 1;
	static int generate = 0;
	static float z0, z1;

	// provide a seed on the first call
	if ( init ) {
		srand48(1);
		init = 0;
	}

	// return z1 if z0 was returned in the previous call
	generate = !generate;
	if ( !generate ) {
		return z1 * sigma + mu;
	}

	// generate number pair (z0, z1), return z0
	float u1 = drand48();
	float u2 = drand48();

	z0 = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * M_PI * u2);
	z1 = sqrtf(-2.0 * logf(u1)) * sinf(2.0 * M_PI * u2);

	return z0 * sigma + mu;
}
