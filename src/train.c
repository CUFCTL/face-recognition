/**
 * @file train.c
 *
 * Train a face database with a set of images.
 */
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include "database.h"

int main(int argc, char **argv)
{
	static struct option long_options[] = {
		{"pca", no_argument, 0,  'p' },
	  {"lda", no_argument, 0,  'l' },
	  {"ica", no_argument, 0,  'i' },
		{"all", no_argument, 0,  'a' },
	  {0,     0,           0,   0  }
	};

	while(opt = getopt_long_only(argc, argv, "", long_options, &options_index) != -1)
	{
		switch (opt)
		{

		}
	}


	const char *TRAINING_SET_PATH = argv[1];
	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";

	database_t *db = db_construct();

	db_train(db, TRAINING_SET_PATH);
	db_save(db, DB_TRAINING_SET, DB_TRAINING_DATA);

	db_destruct(db);

	return 0;
}
