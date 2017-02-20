#!/usr/bin/python
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the face recognition system with a dataset. The dataset should
# have a script of the form "./scripts/create-[dataset].py" that
# can partition the dataset into training and test sets.
import argparse
import os
import random
import shutil
import subprocess
import sys

# parse command-line arguments
parser = argparse.ArgumentParser(epilog="Arguments for C code should be separated by a '--'.")
parser.add_argument("-d", "--dataset", choices=["mnist", "orl", "yale"], required=True, help="name of dataset", dest="DATASET")
parser.add_argument("-t", "--num-train", type=int, help="number of samples per class in training set", metavar="N", dest="NUM_TRAIN")
parser.add_argument("-r", "--num-test", type=int, help="number of samples per class in test set", metavar="N", dest="NUM_TEST")
parser.add_argument("-i", "--num-iter", type=int, required=True, help="number of iterations", metavar="N", dest="NUM_ITER")
parser.add_argument("--run-matlab", action="store_true", help="run MATLAB code", dest="RUN_MATLAB")
parser.add_argument("--run-c", action="store_true", help="run C code", dest="RUN_C")
parser.add_argument("--pca", action="store_true", help="run PCA", dest="PCA")
parser.add_argument("--lda", action="store_true", help="run LDA", dest="LDA")
parser.add_argument("--ica", action="store_true", help="run ICA", dest="ICA")
parser.add_argument("ARGS", nargs=argparse.REMAINDER, help="arguments for C code")

args = parser.parse_args()

# perform some custom validation
if args.NUM_TRAIN is None and args.NUM_TEST is None:
	print "error: you must specify num_train and/or num_test"
	sys.exit(1)

if not args.RUN_MATLAB and not args.RUN_C:
	print "error: you must specify MATLAB code and/or C code"
	sys.exit(1)

if not args.PCA and not args.LDA and not args.ICA:
	print "error: you must specify at least one algorithm (PCA, LDA, ICA)"
	sys.exit(1)

# determine parameters of dataset
if args.DATASET == "mnist":
	CLASS_SIZE = 800
elif args.DATASET == "orl":
	CLASS_SIZE = 10
elif args.DATASET == "yale":
	CLASS_SIZE = 11

if args.NUM_TRAIN is None:
	args.NUM_TRAIN = CLASS_SIZE - args.NUM_TEST
elif args.NUM_TEST is None:
	args.NUM_TEST = CLASS_SIZE - args.NUM_TRAIN

if args.NUM_TRAIN + args.NUM_TEST > CLASS_SIZE:
	print "error: cannot take more than %d per class from %s dataset" % (CLASS_SIZE, args.DATASET)
	sys.exit(1)

# remove '--' from ARGS
if len(args.ARGS) > 0:
	args.ARGS.remove("--")

# add algorithm arguments to ARGS
if args.PCA:
	args.ARGS.append("--pca")

if args.LDA:
	args.ARGS.append("--lda")

if args.ICA:
	args.ARGS.append("--ica")

args.ARGS = " ".join(args.ARGS)

# build face-rec executable
if args.RUN_C:
	print "Building..."
	print
	subprocess.call("make > /dev/null", shell=True)

# perform repeated random testing
for i in xrange(args.NUM_ITER):
	# select a set of random observations
	samples = random.sample(xrange(CLASS_SIZE), args.NUM_TRAIN + args.NUM_TEST)
	samples_train = " ".join([str(s) for s in samples[0:args.NUM_TRAIN]])
	samples_test = " ".join([str(s) for s in samples[args.NUM_TRAIN:]])

	print "TEST %d" % (i + 1)
	print

	# create the training set and test set
	subprocess.call("python scripts/create-sets.py -d %s -t %s -r %s" % (args.DATASET, samples_train, samples_test), shell=True)

	# run the algorithms
	if args.RUN_MATLAB:
		if args.RUN_MATLAB and args.RUN_C:
			print "MATLAB:"

		script = "cd MATLAB; face_rec('../train_images', '../test_images', %d, %d, %d, false); quit" % (args.PCA, args.LDA, args.ICA)
		num_lines = 1 + args.PCA + args.LDA + args.ICA

		subprocess.call("matlab -nojvm -nodisplay -nosplash -r \"%s\" | tail -n %d" % (script, num_lines), shell=True)

	if args.RUN_C:
		if args.RUN_MATLAB and args.RUN_C:
			print
			print "C:"

		subprocess.call("./face-rec --train train_images --test test_images %s" % (args.ARGS), shell=True)

	print
