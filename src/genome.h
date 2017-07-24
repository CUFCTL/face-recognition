/**
 * @file genome.h
 *
 * Interface definitions for RNA-seq data (GTEx dataset).
 */
#ifndef GENOME_H
#define GENOME_H

#include <string>
#include <vector>

class Genome {
private:
	int _gene_count;
	std::vector<float> _expr_lvls;

public:
	Genome();
	~Genome();

	inline int gene_count() const { return this->_gene_count; }
	inline float elem(int i) const { return this->_expr_lvls[i]; }

	void load_rna_seq(const std::string& path);
};

#endif
