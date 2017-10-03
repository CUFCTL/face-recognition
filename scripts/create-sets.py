#!/usr/bin/python
# Create a training set and test set from a dataset
# by removing a set of observations from each class.
import argparse
import os
import random
import shutil
import sys
import datasets

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", choices=["feret", "mnist", "orl", "gtex", "gtex_30", "lab"], required=True, help="name of dataset", dest="DATASET")
parser.add_argument("-t", "--train", type=int, choices=range(1, 100), required=True, help="percentage of training set", metavar="N", dest="TRAIN")
parser.add_argument("-r", "--test", type=int, choices=range(1, 100), required=True, help="percentage of test set", metavar="N", dest="TEST")

args = parser.parse_args()

def get_sub_dirs(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

# perform some custom validation
if args.TRAIN + args.TEST != 100:
	print "error: --train and --test must sum to 100%"
	sys.exit(1)

# determine parameters of dataset
if args.DATASET == "feret":
	dataset = datasets.FERETDataset()
elif args.DATASET == "mnist":
	dataset = datasets.MNISTDataset()
elif args.DATASET == "orl":
	dataset = datasets.ORLDataset()
elif args.DATASET == "gtex":
	subs = get_sub_dirs('datasets/GTEx_Data')
	subs.sort()
	dataset = datasets.GTEXDataset(subs)
elif args.DATASET == "gtex_30":
	subs = get_sub_dirs('datasets/GTEx_Data_30')
	subs.sort()
	dataset = datasets.GTEXDataset30(subs)
elif args.DATASET == "lab":
	subs = get_sub_dirs('datasets/lab_faces')
	dataset = datasets.LABDataset(subs)

# initialize the training set and test set
TRAIN_PATH = "train_data"
TEST_PATH = "test_data"

if os.path.isdir(TRAIN_PATH):
	shutil.rmtree(TRAIN_PATH)

if os.path.isdir(TEST_PATH):
	shutil.rmtree(TEST_PATH)

os.mkdir(TRAIN_PATH, 0775)
os.mkdir(TEST_PATH, 0775)

# partition the data set into training set and test set
for i in xrange(dataset.NUM_CLASSES):
	class_path = dataset.get_class_path(dataset.PATH, i)
	class_files = dataset.get_class_files(dataset.PATH, i)

	num_train = int(len(class_files) * args.TRAIN / 100)
	samples = random.sample(xrange(len(class_files)), len(class_files))
	samples_train = samples[0:num_train]
	samples_test = samples[num_train:]

	# move training images
	train_files = [class_files[j] for j in samples_train]

	for f in train_files:
		filename = dataset.get_dst_filename(i, f)
		src = os.path.join(class_path, f)
		dst = os.path.join(TRAIN_PATH, filename)
		shutil.copy(src, dst)

	# move test images
	test_files = [class_files[j] for j in samples_test]

	for f in test_files:
		filename = dataset.get_dst_filename(i, f)
		src = os.path.join(class_path, f)
		dst = os.path.join(TEST_PATH, filename)
		shutil.copy(src, dst)
