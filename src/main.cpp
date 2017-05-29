/**
 * @file main.cpp
 *
 * User interface to the face recognition system.
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "bayes.h"
#include "dataset.h"
#include "ica.h"
#include "identity.h"
#include "knn.h"
#include "lda.h"
#include "logger.h"
#include "model.h"
#include "pca.h"
#include "timer.h"

#ifdef __NVCC__
	#include "magma_v2.h"
#endif

typedef enum {
	FEATURE_NONE,
	FEATURE_PCA,
	FEATURE_LDA,
	FEATURE_ICA
} feature_type_t;

typedef enum {
	CLASSIFIER_NONE,
	CLASSIFIER_KNN,
	CLASSIFIER_BAYES
} classifier_type_t;

typedef enum {
	OPTION_LOGLEVEL,
	OPTION_TRAIN,
	OPTION_TEST,
	OPTION_STREAM,
	OPTION_PCA,
	OPTION_LDA,
	OPTION_ICA,
	OPTION_KNN,
	OPTION_BAYES,
	OPTION_PCA_N1,
	OPTION_LDA_N1,
	OPTION_LDA_N2,
	OPTION_ICA_N1,
	OPTION_ICA_N2,
	OPTION_ICA_NONL,
	OPTION_ICA_MAX_ITERATIONS,
	OPTION_ICA_EPSILON,
	OPTION_KNN_K,
	OPTION_KNN_DIST,
	OPTION_UNKNOWN = '?'
} option_t;

typedef struct {
	bool train;
	bool test;
	bool stream;
	char *path_train;
	char *path_test;
	feature_type_t feature_type;
	classifier_type_t classifier_type;
	int pca_n1;
	int lda_n1;
	int lda_n2;
	int ica_n1;
	int ica_n2;
	ica_nonl_t ica_nonl;
	int ica_max_iter;
	precision_t ica_eps;
	int knn_k;
	dist_func_t knn_dist;
} optarg_t;

void print_usage()
{
	fprintf(stderr,
		"Usage: ./face-rec [options]\n"
		"\n"
		"Options:\n"
		"  --loglevel LEVEL   set the log level ([1]=info, 2=verbose, 3=debug)\n"
		"  --train DIRECTORY  train a model with a training set\n"
		"  --test DIRECTORY   perform recognition on a test set\n"
		"  --stream           perform recognition on an input stream\n"
		"  --pca              use PCA for feature extraction\n"
		"  --lda              use LDA for feature extraction\n"
		"  --ica              use ICA for feature extraction\n"
		"  --knn              use the kNN classifier (default)\n"
		"  --bayes            use the Bayes classifier\n"
		"\n"
		"Hyperparameters:\n"
		"  --pca_n1 N              (PCA) number of principal components to compute\n"
		"  --lda_n1 N              (LDA) number of principal components to compute\n"
		"  --lda_n2 N              (LDA) number of Fisherfaces to compute\n"
		"  --ica_n1 N              (ICA) number of principal components to compute\n"
		"  --ica_n2 N              (ICA) number of independent components to estimate\n"
		"  --ica_nonl [nonl]       (ICA) nonlinearity function to use (pow3, tanh, gauss)\n"
		"  --ica_max_iterations N  (ICA) maximum iterations\n"
		"  --ica_epsilon X         (ICA) convergence threshold for w\n"
		"  --knn_k N               (kNN) number of nearest neighbors to use\n"
		"  --knn_dist [dist]       (kNN) distance function to use (L1, L2, COS)\n"
	);
}

/**
 * Parse a distance function from a name.
 *
 * @param name
 */
dist_func_t parse_dist_func(const std::string& name)
{
	if ( name == "COS" ) {
		return m_dist_COS;
	}
	else if ( name == "L1" ) {
		return m_dist_L1;
	}
	else if ( name == "L2" ) {
		return m_dist_L2;
	}

	return nullptr;
}

/**
 * Parse a nonlinearity function from a name.
 *
 * @param name
 */
ica_nonl_t parse_nonl_func(const std::string& name)
{
	if ( name == "pow3" ) {
		return ICA_NONL_POW3;
	}
	else if ( name == "tanh" ) {
		return ICA_NONL_TANH;
	}
	else if ( name == "gauss" ) {
		return ICA_NONL_GAUSS;
	}

	return ICA_NONL_NONE;
}

int main(int argc, char **argv)
{
#ifdef __NVCC__
	magma_int_t stat = magma_init();
	assert(stat == MAGMA_SUCCESS);
#endif

	const char *MODEL_FNAME = "./model.dat";

	optarg_t args = {
		false,
		false,
		false,
		nullptr,
		nullptr,
		FEATURE_NONE,
		CLASSIFIER_KNN,
		-1,
		-1, -1,
		-1, -1, ICA_NONL_POW3, 1000, 0.0001f,
		1, m_dist_L2
	};

	struct option long_options[] = {
		{ "loglevel", required_argument, 0, OPTION_LOGLEVEL },
		{ "train", required_argument, 0, OPTION_TRAIN },
		{ "test", required_argument, 0, OPTION_TEST },
		{ "stream", no_argument, 0, OPTION_STREAM },
		{ "pca", no_argument, 0, OPTION_PCA },
		{ "lda", no_argument, 0, OPTION_LDA },
		{ "ica", no_argument, 0, OPTION_ICA },
		{ "knn", no_argument, 0, OPTION_KNN },
		{ "bayes", no_argument, 0, OPTION_BAYES },
		{ "pca_n1", required_argument, 0, OPTION_PCA_N1 },
		{ "lda_n1", required_argument, 0, OPTION_LDA_N1 },
		{ "lda_n2", required_argument, 0, OPTION_LDA_N2 },
		{ "ica_n1", required_argument, 0, OPTION_ICA_N1 },
		{ "ica_n2", required_argument, 0, OPTION_ICA_N2 },
		{ "ica_nonl", required_argument, 0, OPTION_ICA_NONL },
		{ "ica_max_iterations", required_argument, 0, OPTION_ICA_MAX_ITERATIONS },
		{ "ica_epsilon", required_argument, 0, OPTION_ICA_EPSILON },
		{ "knn_k", required_argument, 0, OPTION_KNN_K },
		{ "knn_dist", required_argument, 0, OPTION_KNN_DIST },
		{ 0, 0, 0, 0 }
	};

	// parse command-line arguments
	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, nullptr)) != -1 ) {
		switch ( opt ) {
		case OPTION_LOGLEVEL:
			LOGLEVEL = (logger_level_t) atoi(optarg);
			break;
		case OPTION_TRAIN:
			args.train = true;
			args.path_train = optarg;
			break;
		case OPTION_TEST:
			args.test = true;
			args.path_test = optarg;
			break;
		case OPTION_STREAM:
			args.stream = true;
			break;
		case OPTION_PCA:
			args.feature_type = FEATURE_PCA;
			break;
		case OPTION_LDA:
			args.feature_type = FEATURE_LDA;
			break;
		case OPTION_ICA:
			args.feature_type = FEATURE_ICA;
			break;
		case OPTION_KNN:
			args.classifier_type = CLASSIFIER_KNN;
			break;
		case OPTION_BAYES:
			args.classifier_type = CLASSIFIER_BAYES;
			break;
		case OPTION_PCA_N1:
			args.pca_n1 = atoi(optarg);
			break;
		case OPTION_LDA_N1:
			args.lda_n1 = atoi(optarg);
			break;
		case OPTION_LDA_N2:
			args.lda_n2 = atoi(optarg);
			break;
		case OPTION_ICA_N1:
			args.ica_n1 = atoi(optarg);
			break;
		case OPTION_ICA_N2:
			args.ica_n2 = atoi(optarg);
			break;
		case OPTION_ICA_NONL:
			args.ica_nonl = parse_nonl_func(optarg);
			break;
		case OPTION_ICA_MAX_ITERATIONS:
			args.ica_max_iter = atoi(optarg);
			break;
		case OPTION_ICA_EPSILON:
			args.ica_eps = atof(optarg);
			break;
		case OPTION_KNN_K:
			args.knn_k = atoi(optarg);
			break;
		case OPTION_KNN_DIST:
			args.knn_dist = parse_dist_func(optarg);
			break;
		case OPTION_UNKNOWN:
			print_usage();
			exit(1);
		}
	}

	// validate arguments
	if ( !args.train && !args.test ) {
		print_usage();
		exit(1);
	}

	if ( args.knn_dist == nullptr ) {
		fprintf(stderr, "error: --knn_dist must be L1 | L2 | COS\n");
		exit(1);
	}

	if ( args.ica_nonl == ICA_NONL_NONE ) {
		fprintf(stderr, "error: --ica_nonl must be pow3 | tanh | gauss\n");
		exit(1);
	}

	// initialize feature layer
	FeatureLayer *feature;

	if ( args.feature_type == FEATURE_NONE ) {
		feature = new IdentityLayer();
	}
	else if ( args.feature_type == FEATURE_PCA ) {
		feature = new PCALayer(args.pca_n1);
	}
	else if ( args.feature_type == FEATURE_LDA ) {
		feature = new LDALayer(args.lda_n1, args.lda_n2);
	}
	else if ( args.feature_type == FEATURE_ICA ) {
		feature = new ICALayer(
			args.ica_n1,
			args.ica_n2,
			args.ica_nonl,
			args.ica_max_iter,
			args.ica_eps
		);
	}

	// initialize classifier layer
	ClassifierLayer *classifier;

	if ( args.classifier_type == CLASSIFIER_KNN ) {
		classifier = new KNNLayer(args.knn_k, args.knn_dist);
	}
	else if ( args.classifier_type == CLASSIFIER_BAYES ) {
		classifier = new BayesLayer();
	}

	// initialize model
	Model model(feature, classifier);

	// run the face recognition system
	if ( args.train ) {
		Dataset train_set(args.path_train);

		model.train(train_set);
	}
	else {
		model.load(MODEL_FNAME);
	}

	if ( args.test && args.stream ) {
		char END = '0';
		char READ = '1';

		while ( 1 ) {
			char c = getchar();

			if ( c == END ) {
				break;
			}
			else if ( c == READ ) {
				Dataset test_set(args.path_test);

				std::vector<data_label_t> Y_pred = model.predict(test_set);
				model.validate(test_set, Y_pred);
			}
		}
	}
	else if ( args.test ) {
		Dataset test_set(args.path_test);

		std::vector<data_label_t> Y_pred = model.predict(test_set);
		model.validate(test_set, Y_pred);
	}
	else {
		model.save(MODEL_FNAME);
	}

	timer_print();

	model.print_stats();

#ifdef __NVCC__
	stat = magma_finalize();
	assert(stat == MAGMA_SUCCESS);
#endif

	return 0;
}
