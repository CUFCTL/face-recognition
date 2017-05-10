/**
 * @file math_utils.cpp
 *
 * Library of helpful math functions.
 */
#include <math.h>
#include <stdlib.h>
#include "math_utils.h"

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
 * Compute the second power (square) of a number.
 *
 * @param x
 * @return x^2
 */
float pow2(float x)
{
    return powf(x, 2);
}

/**
 * Compute the third power (cube) of a number.
 *
 * @param x
 * @return x^3
 */
float pow3(float x)
{
    return powf(x, 3);
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

/**
 * Compute the hyperbolic secant of a number.
 *
 * @param x
 * @return sech(x)
 */
float sechf(float x)
{
    return 1.0f / coshf(x);
}
