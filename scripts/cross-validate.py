#!/usr/bin/python
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the face recognition system with a dataset. The dataset should
# have a script of the form "./scripts/create-[dataset].py" that
# can partition the dataset into training and test sets.
import argparse
import os
import shutil
import subprocess
import sys
import datasets

# parse command-line arguments
parser = argparse.ArgumentParser(epilog="Arguments for C code should be separated by a '--'.")
parser.add_argument("-d", "--dataset", choices=["feret", "mnist", "orl"], required=True, help="name of dataset", dest="DATASET")
parser.add_argument("-t", "--train", type=int, choices=range(1, 100), required=True, help="percentage of training set", metavar="N", dest="TRAIN")
parser.add_argument("-r", "--test", type=int, choices=range(1, 100), required=True, help="percentage of test set", metavar="N", dest="TEST")
parser.add_argument("-i", "--num-iter", type=int, required=True, help="number of iterations", metavar="N", dest="NUM_ITER")
parser.add_argument("--pca", action="store_true", help="run PCA", dest="PCA")
parser.add_argument("--lda", action="store_true", help="run LDA", dest="LDA")
parser.add_argument("--ica", action="store_true", help="run ICA", dest="ICA")
parser.add_argument("ARGS", nargs=argparse.REMAINDER, help="arguments for C code")

args = parser.parse_args()

# perform some custom validation
if args.TRAIN + args.TEST != 100:
	print "error: --train and --test must sum to 100%"
	sys.exit(1)

if not args.PCA and not args.LDA and not args.ICA:
	print "error: you must specify at least one algorithm (PCA, LDA, ICA)"
	sys.exit(1)

# check for face-rec executable
if not os.path.isfile("./face-rec"):
	print "error: ./face-rec not found"
	sys.exit(1)

# determine parameters of dataset
if args.DATASET == "feret":
	dataset = datasets.FERETDataset()
elif args.DATASET == "mnist":
	dataset = datasets.MNISTDataset()
elif args.DATASET == "orl":
	dataset = datasets.ORLDataset()

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

# construct subprocess args
args_dataset = ["python", "scripts/create-sets.py", "-d", args.DATASET, "-t", str(args.TRAIN), "-r", str(args.TEST)]

args_c = ["./face-rec", "--train", "train_images", "--test", "test_images"]
args_c.extend(args.ARGS);

# construct results object
results = []

# perform repeated random testing
for i in xrange(args.NUM_ITER):
	print "TEST %d" % (i + 1)

	# create the training set and test set
	subprocess.call(args_dataset)

	# run the system
	p = subprocess.Popen(args_c, stdout=subprocess.PIPE)
	p.wait()

	data = p.stdout.readline().split()
	data = [float(d) for d in data]

	if len(results) == 0:
		results = [0 for d in data]

	results = [(results[i] + data[i]) for i in xrange(len(data))]

# average results
results = [(r / args.NUM_ITER) for r in results]

print
print "".join(["{0:12.3f}".format(r) for r in results])
