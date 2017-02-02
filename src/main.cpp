/**
 * @file main.cpp
 *
 * User interface to the face recognition system.
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "database.h"
#include "logger.h"
#include "timer.h"

logger_level_t LOGLEVEL = LL_INFO;
int TIMING = 0;

typedef enum {
	OPTION_LOGLEVEL,
	OPTION_TIMING,
	OPTION_TRAIN,
	OPTION_TEST,
	OPTION_PCA,
	OPTION_LDA,
	OPTION_ICA,
	OPTION_ALL,
	OPTION_PCA_N1,
	OPTION_PCA_DIST,
	OPTION_LDA_N1,
	OPTION_LDA_N2,
	OPTION_LDA_DIST,
	OPTION_ICA_NUM_IC,
	OPTION_ICA_MAX_ITERATIONS,
	OPTION_ICA_EPSILON,
	OPTION_ICA_DIST,
	OPTION_KNN_K,
	OPTION_UNKNOWN = '?'
} optarg_t;

void print_usage()
{
	fprintf(stderr,
		"Usage: ./face-rec [options]\n"
		"\n"
		"Options:\n"
		"  --loglevel LEVEL   set the log level ([1]=info, 2=verbose, 3=debug)\n"
		"  --timing           print timing information\n"
		"  --train DIRECTORY  train a database with a training set\n"
		"  --test DIRECTORY   perform recognition on a test set\n"
		"  --pca              run PCA\n"
		"  --lda              run LDA\n"
		"  --ica              run ICA\n"
		"  --all              run all algorithms (PCA, LDA, ICA)\n"
		"\n"
		"Hyperparameters:\n"
		"  --pca_n1 N              (PCA) number of columns in W_pca to use\n"
		"  --pca_dist COS|L1|L2    (PCA) distance function to use\n"
		"  --lda_n1 N              (LDA) number of columns in W_pca to use\n"
		"  --lda_n2 N              (LDA) number of columns in W_fld to use\n"
		"  --lda_dist COS|L1|L2    (LDA) distance function to use\n"
		"  --ica_num_ic N          (ICA) number of independent components to estimate\n"
		"  --ica_max_iterations N  (ICA) maximum iterations\n"
		"  --ica_epsilon X         (ICA) convergence threshold for w\n"
		"  --ica_dist COS|L1|L2    (ICA) distance function to use\n"
		"  --knn_k N               (kNN) number of nearest neighbors to use\n"
	);
}

/**
 * Parse a distance function from a name.
 *
 * @param name
 * @return pointer to corresponding distance function
 */
dist_func_t parse_dist_func(const char *name)
{
	if ( strcmp(name, "COS") == 0 ) {
		return m_dist_COS;
	}
	else if ( strcmp(name, "L1") == 0 ) {
		return m_dist_L1;
	}
	else if ( strcmp(name, "L2") == 0 ) {
		return m_dist_L2;
	}

	return NULL;
}

int main(int argc, char **argv)
{
	const char *DB_DATA = "./database.dat";

	bool arg_train = false;
	bool arg_test = false;
	bool arg_pca = false;
	bool arg_lda = false;
	bool arg_ica = false;

	db_params_t db_params = {
		{ -1, m_dist_L2 },
		{ -1, -1, m_dist_L2 },
		{ -1, 1000, 0.0001f, m_dist_COS },
		{ 1 }
	};

	char *path_train_set = NULL;
	char *path_test_set = NULL;

	struct option long_options[] = {
		{ "loglevel", required_argument, 0, OPTION_LOGLEVEL },
		{ "timing", no_argument, 0, OPTION_TIMING },
		{ "train", required_argument, 0, OPTION_TRAIN },
		{ "test", required_argument, 0, OPTION_TEST },
		{ "pca", no_argument, 0, OPTION_PCA },
		{ "lda", no_argument, 0, OPTION_LDA },
		{ "ica", no_argument, 0, OPTION_ICA },
		{ "all", no_argument, 0, OPTION_ALL },
		{ "pca_n1", required_argument, 0, OPTION_PCA_N1 },
		{ "pca_dist", required_argument, 0, OPTION_PCA_DIST },
		{ "lda_n1", required_argument, 0, OPTION_LDA_N1 },
		{ "lda_n2", required_argument, 0, OPTION_LDA_N2 },
		{ "lda_dist", required_argument, 0, OPTION_LDA_DIST },
		{ "ica_num_ic", required_argument, 0, OPTION_ICA_NUM_IC },
		{ "ica_max_iterations", required_argument, 0, OPTION_ICA_MAX_ITERATIONS },
		{ "ica_epsilon", required_argument, 0, OPTION_ICA_EPSILON },
		{ "ica_dist", required_argument, 0, OPTION_ICA_DIST },
		{ "knn_k", required_argument, 0, OPTION_KNN_K },
		{ 0, 0, 0, 0 }
	};

	// parse command-line arguments
	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case OPTION_LOGLEVEL:
			LOGLEVEL = (logger_level_t) atoi(optarg);
			break;
		case OPTION_TIMING:
			TIMING = 1;
			break;
		case OPTION_TRAIN:
			arg_train = true;
			path_train_set = optarg;
			break;
		case OPTION_TEST:
			arg_test = true;
			path_test_set = optarg;
			break;
		case OPTION_PCA:
			arg_pca = true;
			break;
		case OPTION_LDA:
			arg_lda = true;
			break;
		case OPTION_ICA:
			arg_ica = true;
			break;
		case OPTION_ALL:
			arg_pca = true;
			arg_lda = true;
			arg_ica = true;
			break;
		case OPTION_PCA_N1:
			db_params.pca.n1 = atoi(optarg);
			break;
		case OPTION_PCA_DIST:
			db_params.pca.dist = parse_dist_func(optarg);
			break;
		case OPTION_LDA_N1:
			db_params.lda.n1 = atoi(optarg);
			break;
		case OPTION_LDA_N2:
			db_params.lda.n2 = atoi(optarg);
			break;
		case OPTION_LDA_DIST:
			db_params.lda.dist = parse_dist_func(optarg);
			break;
		case OPTION_ICA_NUM_IC:
			db_params.ica.num_ic = atoi(optarg);
			break;
		case OPTION_ICA_MAX_ITERATIONS:
			db_params.ica.max_iterations = atoi(optarg);
			break;
		case OPTION_ICA_EPSILON:
			db_params.ica.epsilon = atof(optarg);
			break;
		case OPTION_ICA_DIST:
			db_params.ica.dist = parse_dist_func(optarg);
			break;
		case OPTION_KNN_K:
			db_params.knn.k = atoi(optarg);
			break;
		case OPTION_UNKNOWN:
			print_usage();
			exit(1);
		}
	}

	// validate arguments
	if ( !arg_train && !arg_test ) {
		print_usage();
		exit(1);
	}

	if ( db_params.pca.dist == NULL || db_params.lda.dist == NULL || db_params.ica.dist == NULL ) {
		fprintf(stderr, "error: dist function must be L1 | L2 | COS\n");
		exit(1);
	}

	// run the face recognition system
	database_t *db = db_construct(arg_pca, arg_lda, arg_ica, db_params);

	if ( arg_train ) {
		db_train(db, path_train_set);
	}
	else {
		db_load(db, DB_DATA);
	}

	if ( arg_test ) {
		db_recognize(db, path_test_set);
	}
	else {
		db_save(db, DB_DATA);
	}

	db_destruct(db);

	// print timing information
	timer_print();

	return 0;
}
