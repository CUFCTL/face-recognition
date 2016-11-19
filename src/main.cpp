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
#include "timing.h"

int VERBOSE = 0;
int TIMING = 0;

void print_usage()
{
	fprintf(stderr,
		"Usage: ./face-rec [options]\n"
		"\n"
		"Options:\n"
		"  --verbose          enable verbose output\n"
		"  --timing           print timing information\n"
		"  --train DIRECTORY  create a database from a training set\n"
		"  --rec DIRECTORY    test a set of images against a database\n"
		"  --pca              run PCA\n"
		"  --lda              run LDA\n"
		"  --ica              run ICA\n"
		"  --all              run all algorithms (PCA, LDA, ICA)\n"
		"  --lda1             LDA parameter #1\n"
		"  --lda2             LDA parameter #2\n"
	);
}

int main(int argc, char **argv)
{
	const char *DB_ENTRIES = "./db_training_set.dat";
	const char *DB_DATA = "./db_training_data.dat";

	int arg_train = 0;
	int arg_recognize = 0;
	int arg_pca = 0;
	int arg_lda = 0;
	int arg_ica = 0;

	int n_opt1 = -1;
	int n_opt2 = -1;

	char *path_train_set = NULL;
	char *path_test_set = NULL;

	struct option long_options[] = {
		{ "verbose", no_argument, 0, 'v' },
		{ "timing", no_argument, 0, 's' },
		{ "train", required_argument, 0, 't' },
		{ "rec", required_argument, 0, 'r' },
		{ "pca", no_argument, 0, 'p' },
		{ "lda", no_argument, 0, 'l' },
		{ "ica", no_argument, 0, 'i' },
		{ "all", no_argument, 0, 'a' },
		{ "lda1", required_argument, 0, 'm' },
		{ "lda2", required_argument, 0, 'n' },
		{ 0, 0, 0, 0 }
	};

	// parse command-line arguments
	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case 'v':
			VERBOSE = 1;
			break;
		case 's':
			TIMING = 1;
			break;
		case 't':
			arg_train = 1;
			path_train_set = optarg;
			break;
		case 'r':
			arg_recognize = 1;
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
		case 'm':
			n_opt1 = atoi(optarg);
			break;
		case 'n':
			n_opt2 = atoi(optarg);
			break;
		case '?':
			print_usage();
			exit(1);
		}
	}

	// validate arguments
	if ( !arg_train && !arg_recognize ) {
		print_usage();
		exit(1);
	}

	// run the face recognition system
	database_t *db = db_construct(arg_pca, arg_lda, arg_ica);

	if ( arg_train && arg_recognize ) {
		timing_start("Train Database");
		db_train(db, path_train_set, n_opt1, n_opt2);
		timing_end("Train Database");

		timing_start("Recognize Images");
		db_recognize(db, path_test_set);
		timing_end("Recognize Images");
	}
	else if ( arg_train ) {
		timing_start("Train Database");
		db_train(db, path_train_set, n_opt1, n_opt2);
		timing_end("Train Database");

		db_save(db, DB_ENTRIES, DB_DATA);
	}
	else if ( arg_recognize ) {
		db_load(db, DB_ENTRIES, DB_DATA);

		timing_start("Train Database");
		db_recognize(db, path_test_set);
		timing_end("Train Database");
	}

	db_destruct(db);

	timing_print();
	return 0;
}
