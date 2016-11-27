/**
 * @file main.c
 *
 * User interface to the face recognition system.
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "database.h"
#include "logger.h"
#include "timer.h"

logger_level_t LOGLEVEL = LL_INFO;
int TIMING = 0;

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
		"  --lda_n1 N              (LDA) number of columns in W_pca to use\n"
		"  --lda_n2 N              (LDA) number of columns in W_fld to use\n"
		"  --ica_num_ic N          (ICA) number of independent components to estimate\n"
		"  --ica_max_iterations N  (ICA) maximum iterations\n"
		"  --ica_epsilon X         (ICA) convergence threshold for w\n"
	);
}

int main(int argc, char **argv)
{
	const char *DB_DATA = "./database.dat";

	int arg_train = 0;
	int arg_test = 0;
	int arg_pca = 0;
	int arg_lda = 0;
	int arg_ica = 0;

	db_params_t db_params = {
		-1,
		-1, -1,
		-1, 1000, 0.0001f
	};

	char *path_train_set = NULL;
	char *path_test_set = NULL;

	struct option long_options[] = {
		{ "loglevel", required_argument, 0, 'e' },
		{ "timing", no_argument, 0, 's' },
		{ "train", required_argument, 0, 't' },
		{ "test", required_argument, 0, 'r' },
		{ "pca", no_argument, 0, 'p' },
		{ "lda", no_argument, 0, 'l' },
		{ "ica", no_argument, 0, 'i' },
		{ "all", no_argument, 0, 'a' },
		{ "pca_n1", required_argument, 0, '1' },
		{ "lda_n1", required_argument, 0, '2' },
		{ "lda_n2", required_argument, 0, '3' },
		{ "ica_num_ic", required_argument, 0, '4' },
		{ "ica_max_iterations", required_argument, 0, '5' },
		{ "ica_epsilon", required_argument, 0, '6' },
		{ 0, 0, 0, 0 }
	};

	// parse command-line arguments
	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case 'e':
			LOGLEVEL = (logger_level_t) atoi(optarg);
			break;
		case 's':
			TIMING = 1;
			break;
		case 't':
			arg_train = 1;
			path_train_set = optarg;
			break;
		case 'r':
			arg_test = 1;
			path_test_set = optarg;
			break;
		case 'p':
			arg_pca = 1;
			break;
		case 'l':
			arg_lda = 1;
			break;
		case 'i':
			arg_ica = 1;
			break;
		case 'a':
			arg_pca = 1;
			arg_lda = 1;
			arg_ica = 1;
			break;
		case '1':
			db_params.pca_n1 = atoi(optarg);
			break;
		case '2':
			db_params.lda_n1 = atoi(optarg);
			break;
		case '3':
			db_params.lda_n2 = atoi(optarg);
			break;
		case '4':
			db_params.ica_num_ic = atoi(optarg);
			break;
		case '5':
			db_params.ica_max_iterations = atoi(optarg);
			break;
		case '6':
			db_params.ica_epsilon = atof(optarg);
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	// validate arguments
	if ( !arg_train && !arg_test ) {
		print_usage();
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
