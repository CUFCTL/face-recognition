/**
 * @file model.cpp
 *
 * Implementation of the model type.
 */
#include <iomanip>
#include "image.h"
#include "logger.h"
#include "model.h"
#include "timer.h"

/**
 * Construct a model.
 *
 * @param feature
 * @param classifier
 */
Model::Model(FeatureLayer *feature, ClassifierLayer *classifier)
{
	// initialize layers
	this->_feature = feature;
	this->_classifier = classifier;

	// initialize stats
	this->_stats.accuracy = 0.0f;
	this->_stats.train_time = 0.0f;
	this->_stats.test_time = 0.0f;

	// log hyperparameters
	log(LL_VERBOSE, "Hyperparameters");

	this->_feature->print();
	this->_classifier->print();

	log(LL_VERBOSE, "");
}

/**
 * Destruct a model.
 */
Model::~Model()
{
	delete this->_feature;
	delete this->_classifier;
}

/**
 * Save a model to a file.
 *
 * @param path
 */
void Model::save(const std::string& path)
{
	std::ofstream file(path, std::ofstream::out);

	this->_train_set.save(file);
	this->_mean.save(file);
	this->_feature->save(file);
	this->_P.save(file);

	file.close();
}

/**
 * Load a model from a file.
 *
 * @param path
 */
void Model::load(const std::string& path)
{
	std::ifstream file(path, std::ifstream::in);

	this->_train_set.load(file);
	this->_mean.load(file);
	this->_feature->load(file);
	this->_P.load(file);

	file.close();
}

/**
 * Perform training on a training set.
 *
 * @param train_set
 */
void Model::train(const Dataset& train_set)
{
	timer_push("Training");

	this->_train_set = train_set;

	log(LL_VERBOSE, "Training set: %d samples, %d classes",
		train_set.entries().size(),
		train_set.labels().size());

	// get data matrix X
	Matrix X = train_set.load_data();

	// subtract mean from X
	this->_mean = X.mean_column("m");

	X.subtract_columns(this->_mean);

	// project X into feature space
	this->_feature->compute(X, this->_train_set.entries(), this->_train_set.labels().size());
	this->_P = this->_feature->project(X);

	// record training time
	this->_stats.train_time = timer_pop();

	log(LL_VERBOSE, "");
}

/**
 * Perform recognition on a test set.
 *
 * @param test_set
 */
std::vector<DataLabel> Model::predict(const Dataset& test_set)
{
	timer_push("Prediction");

	log(LL_VERBOSE, "Test set: %d samples, %d classes",
		test_set.entries().size(),
		test_set.labels().size());

	// compute projected test images
	Matrix X_test = test_set.load_data();
	X_test.subtract_columns(this->_mean);

	Matrix P_test = this->_feature->project(X_test);

	// compute predicted labels
	std::vector<DataLabel> Y_pred = this->_classifier->predict(
		this->_P,
		this->_train_set.entries(),
		this->_train_set.labels(),
		P_test
	);

	// record predition time
	this->_stats.test_time = timer_pop();

	log(LL_VERBOSE, "");

	return Y_pred;
}

/**
 * Validate a set of predicted labels against the ground truth.
 *
 * @param test_set
 * @param Y_pred
 */
void Model::validate(const Dataset& test_set, const std::vector<DataLabel>& Y_pred)
{
	// compute accuracy
	int num_correct = 0;

	for ( size_t i = 0; i < test_set.entries().size(); i++ ) {
		if ( Y_pred[i] == test_set.entries()[i].label ) {
			num_correct++;
		}
	}

	this->_stats.accuracy = 100.0f * num_correct / test_set.entries().size();

	// print results
	log(LL_VERBOSE, "Results");

	for ( size_t i = 0; i < test_set.entries().size(); i++ ) {
		const DataLabel& y_pred = Y_pred[i];
		const DataEntry& entry = test_set.entries()[i];

		const char *s = (y_pred != entry.label)
			? "(!)"
			: "";

		log(LL_VERBOSE, "%-12s -> %-4s %s", entry.name.c_str(), y_pred.c_str(), s);
	}

	log(LL_VERBOSE, "%d / %d matched, %.2f%%", num_correct, test_set.entries().size(), this->_stats.accuracy);
	log(LL_VERBOSE, "");
}

/**
 * Print a model's performance / accuracy stats.
 */
void Model::print_stats()
{
	std::cout
		<< std::setw(12) << std::setprecision(3) << this->_stats.accuracy
		<< std::setw(12) << std::setprecision(3) << this->_stats.train_time
		<< std::setw(12) << std::setprecision(3) << this->_stats.test_time
		<< "\n";
}
