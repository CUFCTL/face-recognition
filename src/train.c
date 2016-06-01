/**
 * @file train.c
 *
 * Train a face database with a set of images.
 */
#include <stdio.h>
#include "database.h"

int main(int argc, char **argv)
{
	if ( argc != 2 ) {
		fprintf(stderr, "usage: ./train [images-folder]\n");
		return 1;
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
