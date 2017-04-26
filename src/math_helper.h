/**
 * @file math_helper.h
 *
 * Library of helpful math functions.
 */
#ifndef MATH_HELPER_H
#define MATH_HELPER_H

#include <stdlib.h>

int max(int x, int y);
int min(int x, int y);
void * mode(const void *base, size_t nmemb, size_t size, void * (*identity)(const void *));
float pow2(float x);
float pow3(float x);
float rand_normal(float mu, float sigma);
float sechf(float x);

#endif
