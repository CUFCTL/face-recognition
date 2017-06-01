/**
 * @file identity.cpp
 *
 * Implementation of identity feature layer.
 */
#include "identity.h"
#include "logger.h"

/**
 * Compute the features for an identity layer.
 *
 * NOTE: since the identity layer just returns the input,
 * this function returns nullptr in lieu of allocating a
 * large identity matrix.
 *
 * @param X
 * @param y
 * @param c
 */
void IdentityLayer::compute(const Matrix& X, const std::vector<DataEntry>& y, int c)
{
}

/**
 * Project an input matrix into the feature space
 * of the identity layer.
 *
 * @param X
 */
Matrix IdentityLayer::project(const Matrix& X)
{
	return X;
}

/**
 * Save an identity layer to a file.
 *
 * @param file
 */
void IdentityLayer::save(std::ofstream& file)
{
}

/**
 * Load an identity layer from a file.
 *
 * @param file
 */
void IdentityLayer::load(std::ifstream& file)
{
}

/**
 * Print information about an identity layer.
 */
void IdentityLayer::print()
{
	log(LL_VERBOSE, "Identity");
}
