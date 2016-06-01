/**
 * @file recognize.c
 *
 * Test a set of images against a face database.
 */
#include <stdio.h>
#include "database.h"

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./recognize [images-folder]\n");
		return 1;
	}

	const char *DB_TRAINING_SET = "./db_training_set.dat";
	const char *DB_TRAINING_DATA = "./db_training_data.dat";
	const char *TEST_SET_PATH = argv[1];

	database_t *db = db_construct();

	db_load(db, DB_TRAINING_SET, DB_TRAINING_DATA);
	db_recognize(db, TEST_SET_PATH);

	db_destruct(db);

	return 0;
}
