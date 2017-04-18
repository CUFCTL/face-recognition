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
#include "logger.h"
#include "model.h"
#include "timer.h"

logger_level_t LOGLEVEL = LL_INFO;
int TIMING = 0;

typedef enum {
	OPTION_LOGLEVEL,
	OPTION_TIMING,
	OPTION_TRAIN,
	OPTION_TEST,
	OPTION_STREAM,
	OPTION_PCA,
	OPTION_LDA,
	OPTION_ICA,
	OPTION_PCA_N1,
	OPTION_LDA_N1,
	OPTION_LDA_N2,
	OPTION_ICA_N1,
	OPTION_ICA_N2,
	OPTION_ICA_MAX_ITERATIONS,
	OPTION_ICA_EPSILON,
	OPTION_KNN_K,
	OPTION_KNN_DIST,
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
		"  --train DIRECTORY  train a model with a training set\n"
		"  --test DIRECTORY   perform recognition on a test set\n"
		"  --stream           perform recognition on an input stream\n"
		"  --pca              run PCA\n"
		"  --lda              run LDA\n"
		"  --ica              run ICA\n"
		"\n"
		"Hyperparameters:\n"
		"  --pca_n1 N              (PCA) number of principal components to compute\n"
		"  --lda_n1 N              (LDA) number of principal components to compute\n"
		"  --lda_n2 N              (LDA) number of Fisherfaces to compute\n"
		"  --ica_n1 N              (ICA) number of principal components to compute\n"
		"  --ica_n2 N              (ICA) number of independent components to estimate\n"
		"  --ica_max_iterations N  (ICA) maximum iterations\n"
		"  --ica_epsilon X         (ICA) convergence threshold for w\n"
		"  --knn_k N               (kNN) number of nearest neighbors to use\n"
		"  --knn_dist COS|L1|L2    (kNN) distance function to use\n"
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
	const char *MODEL_FNAME = "./model.dat";

	bool arg_train = false;
	bool arg_test = false;
	bool arg_stream = false;

	feature_type_t feature_type = FEATURE_NONE;
	classifier_type_t classifier_type = CLASSIFIER_KNN;
	model_params_t model_params = {
		{ -1 },
		{ -1, -1 },
		{ -1, -1, 1000, 0.0001f },
		{ 1, m_dist_L2 }
	};

	char *path_train = NULL;
	char *path_test = NULL;

	struct option long_options[] = {
		{ "loglevel", required_argument, 0, OPTION_LOGLEVEL },
		{ "timing", no_argument, 0, OPTION_TIMING },
		{ "train", required_argument, 0, OPTION_TRAIN },
		{ "test", required_argument, 0, OPTION_TEST },
		{ "stream", no_argument, 0, OPTION_STREAM },
		{ "pca", no_argument, 0, OPTION_PCA },
		{ "lda", no_argument, 0, OPTION_LDA },
		{ "ica", no_argument, 0, OPTION_ICA },
		{ "pca_n1", required_argument, 0, OPTION_PCA_N1 },
		{ "lda_n1", required_argument, 0, OPTION_LDA_N1 },
		{ "lda_n2", required_argument, 0, OPTION_LDA_N2 },
		{ "ica_n1", required_argument, 0, OPTION_ICA_N1 },
		{ "ica_n2", required_argument, 0, OPTION_ICA_N2 },
		{ "ica_max_iterations", required_argument, 0, OPTION_ICA_MAX_ITERATIONS },
		{ "ica_epsilon", required_argument, 0, OPTION_ICA_EPSILON },
		{ "knn_k", required_argument, 0, OPTION_KNN_K },
		{ "knn_dist", required_argument, 0, OPTION_KNN_DIST },
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
			path_train = optarg;
			break;
		case OPTION_TEST:
			arg_test = true;
			path_test = optarg;
			break;
		case OPTION_STREAM:
			arg_stream = true;
			break;
		case OPTION_PCA:
			feature_type = FEATURE_PCA;
			break;
		case OPTION_LDA:
			feature_type = FEATURE_LDA;
			break;
		case OPTION_ICA:
			feature_type = FEATURE_ICA;
			break;
		case OPTION_PCA_N1:
			model_params.pca.n1 = atoi(optarg);
			break;
		case OPTION_LDA_N1:
			model_params.lda.n1 = atoi(optarg);
			break;
		case OPTION_LDA_N2:
			model_params.lda.n2 = atoi(optarg);
			break;
		case OPTION_ICA_N1:
			model_params.ica.n1 = atoi(optarg);
			break;
		case OPTION_ICA_N2:
			model_params.ica.n2 = atoi(optarg);
			break;
		case OPTION_ICA_MAX_ITERATIONS:
			model_params.ica.max_iterations = atoi(optarg);
			break;
		case OPTION_ICA_EPSILON:
			model_params.ica.epsilon = atof(optarg);
			break;
		case OPTION_KNN_K:
			model_params.knn.k = atoi(optarg);
			break;
		case OPTION_KNN_DIST:
			model_params.knn.dist = parse_dist_func(optarg);
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

	if ( model_params.knn.dist == NULL ) {
		fprintf(stderr, "error: dist function must be L1 | L2 | COS\n");
		exit(1);
	}

	// run the face recognition system
	model_t *model = model_construct(feature_type, classifier_type, model_params);

	if ( arg_train ) {
		model_train(model, path_train);
	}
	else {
		model_load(model, MODEL_FNAME);
	}

	if ( arg_test && arg_stream ) {
		char END = '0';
		char READ = '1';

		while ( 1 ) {
			char c = getchar();

			if ( c == END ) {
				break;
			}
			else if ( c == READ ) {
				model_predict(model, path_test);
			}
		}
	}
	else if ( arg_test ) {
		image_label_t **pred_labels = model_predict(model, path_test);

		model_validate(model, path_test, pred_labels);

		free(pred_labels);
	}
	else {
		model_save(model, MODEL_FNAME);
	}

	model_destruct(model);

	// print timing information
	timer_print();

	return 0;
}
