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
#include <time.h>

int VERBOSE = 0;

void print_usage()
{
	fprintf(stderr,
		"Usage: ./face-rec [options]\n"
		"\n"
		"Options:\n"
		"  --verbose          enable verbose output\n"
		"  --train DIRECTORY  create a database from a training set\n"
		"  --rec DIRECTORY    test a set of images against a database\n"
		"  --pca              run PCA\n"
		"  --lda              run LDA\n"
		"  --ica              run ICA\n"
		"  --all              run all algorithms (PCA, LDA, ICA)\n"
		"  --lda1             first optional lda argument\n"
		"  --lda2             second optional lda argument\n"
	);
}

int main(int argc, char **argv)
{
	FILE *fp;
	fp = fopen(FP, "a");

	clock_t main_begin = clock();

	const char *DB_ENTRIES = "./db_training_set.dat";
	const char *DB_DATA = "./db_training_data.dat";

	int arg_train = 0;
	int arg_recognize = 0;
	int arg_pca = 0;
	int arg_lda = 0;
	int arg_ica = 0;

	int n_opt1 = 240;
	int n_opt2 = 39;

	char *path_train_set = NULL;
	char *path_test_set = NULL;

	struct option long_options[] = {
		{ "verbose", no_argument, 0, 'v' },
		{ "train", required_argument, 0, 't' },
		{ "rec", required_argument, 0, 'r' },
		{ "pca", no_argument, 0, 'p' },
		{ "lda", no_argument, 0, 'l' },
		{ "ica", no_argument, 0, 'i' },
		{ "all", no_argument, 0, 'a' },
		{ "lda1", required_argument, 0, 'f' },
		{ "lda2", required_argument, 0, 's' },
		{ 0, 0, 0, 0 }
	};

	// parse command-line arguments
	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case 'v':
			VERBOSE = 1;
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
		case 'f':
			n_opt1 = atoi(optarg);
			break;
		case 's':
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
		clock_t trainTimeBegin = clock();
		db_train(db, path_train_set, n_opt1, n_opt2);
		clock_t trainTimeEnd = clock();
		fprintf(fp, "\nTraining time, %.3lf\n", (double)(trainTimeEnd - trainTimeBegin) / CLOCKS_PER_SEC);

		clock_t recTimeBegin = clock();

		db_recognize(db, path_test_set);
		clock_t recTimeEnd = clock();
		fprintf(fp, "\nRecognization time, %.3lf\n", (double)(recTimeEnd - recTimeBegin) / CLOCKS_PER_SEC);
	}
	else if ( arg_train ) {
		db_train(db, path_train_set, n_opt1, n_opt2);
		db_save(db, DB_ENTRIES, DB_DATA);
	}
	else if ( arg_recognize ) {
		db_load(db, DB_ENTRIES, DB_DATA);
		db_recognize(db, path_test_set);
	}

	db_destruct(db);

	clock_t main_end = clock();
	fprintf(fp, "\nMain time, %.3lf\n", (double)(main_end - main_begin) / CLOCKS_PER_SEC);

	fclose(fp);
	return 0;
}
