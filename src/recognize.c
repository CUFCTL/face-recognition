/**
 * @file recognize.c
 *
 * Test a set of images against a face database.
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "database.h"

void PrintUsage();

int main(int argc, char **argv)
{
	int opt, arg_count = 0;
	args_t * args = (args_t *)malloc(sizeof(args));

	static struct option long_options[] = {
			{"pca", no_argument, 0,  'p' },
			{"lda", no_argument, 0,  'l' },
			{"ica", no_argument, 0,  'i' },
			{"all", no_argument, 0,  'a' },
			{0,     0,           0,   0  }
	};

	while((opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1)
	{
		switch (opt)
		{
			case 'p':
				args->pca = 1;
				arg_count++;
				break;
			case 'l':
				args->lda = 1;
				arg_count++;
				break;
			case 'i':
				args->ica = 1;
				arg_count++;
				break;
			case 'a':
				args->pca = 1;
				args->lda = 1;
				args->ica = 1;
				arg_count++;
				break;
			case '?':
				PrintUsage();
				exit(1);
		}
	}

	if (!arg_count)
	{
		args->pca = 1;
		args->lda = 1;
		args->ica = 1;
	}

	const char *TEST_SET_PATH = argv[arg_count + 1];
	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";

	database_t *db = db_construct();

	db_load(db, DB_TRAINING_SET, DB_TRAINING_DATA, args);
	db_recognize(db, TEST_SET_PATH, args);

	db_destruct(db, args);

	return 0;
}


void PrintUsage()
{
	fprintf(stderr, "Invalid arguments. Usage is as follows:\n");
	fprintf(stderr, "-pca   ---> run pca algorithm\n");
	fprintf(stderr, "-lda   ---> run lda algorithm\n");
	fprintf(stderr, "-ica   ---> run ica algorithm\n");
	fprintf(stderr, "-all   ---> run all algorithms\n");
	fprintf(stderr, "You may call the algorithms in any order and in\n");
	fprintf(stderr, "any combination.\n");
}
