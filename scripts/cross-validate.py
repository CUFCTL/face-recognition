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

# parse command-line arguments
parser = argparse.ArgumentParser(epilog="Arguments for C code should be separated by a '--'.")
parser.add_argument("-d", "--dataset", choices=["orl", "yale"], required=True, help="name of dataset", dest="DATASET")
parser.add_argument("-r", "--num-remove", type=int, required=True, help="number of samples to remove from training set", metavar="N", dest="NUM_REMOVE")
parser.add_argument("-i", "--num-iter", type=int, required=True, help="number of iterations", metavar="N", dest="NUM_ITER")
parser.add_argument("--run-matlab", action="store_true", help="run MATLAB code", dest="RUN_MATLAB")
parser.add_argument("--run-c", action="store_true", help="run C code", dest="RUN_C")
parser.add_argument("--pca", action="store_true", help="run PCA", dest="PCA")
parser.add_argument("--lda", action="store_true", help="run LDA", dest="LDA")
parser.add_argument("--ica", action="store_true", help="run ICA", dest="ICA")
parser.add_argument("ARGS", nargs=argparse.REMAINDER, help="arguments for C code")

args = parser.parse_args()

# determine parameters of dataset
if args.DATASET == "orl":
	DB_PATH = "datasets/orl_faces"
	CLASS_SIZE = 10
elif args.DATASETS == "yale":
	DB_PATH = "datasets/yalefaces"
	CLASS_SIZE = 11

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
print "Performing Monte Carlo cross-validation with p=%d and n=%d" % (args.NUM_REMOVE, args.NUM_ITER)

for i in xrange(args.NUM_ITER):
	# select p random observations
	samples = random.sample(xrange(CLASS_SIZE), args.NUM_REMOVE)
	samples = " ".join([str(s) for s in samples])

	print "TEST %d: removing observations (%s) from each class" % (i + 1, samples)
	print

	# create the training set and test set
	subprocess.call("./scripts/create-%s.py -p %s -s %s" % (args.DATASET, DB_PATH, samples), shell=True)

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
