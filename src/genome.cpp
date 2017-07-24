/**
 * @file genome.cpp
 *
 * Implementation of the genome type.
 *
 * The following formats are supported:
 * - RNA-seq data extracted to .txt file 
 */
#include <cctype>
#include <fstream>
#include "genome.h"
#include "logger.h"

/**
 * Construct a genome.
 */ 
Genome::Genome()
{
	this->_gene_count = 0;
}

/**
 * Destruct a genome.
 */
Genome::~Genome()
{

}

/**
 * Load RNA-seq data from a .txt file
 *
 * @param path
 */
void Genome::load_rna_seq(const std::string& path)
{
	float temp = 0;
	std::ifstream file (path, std::ifstream::in);

	// initialize gene expression level vector
	while (file >> temp)
	{
		this->_expr_lvls.push_back(temp);
	}

	this->_gene_count = _expr_lvls.size();

	file.close();
}