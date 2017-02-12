#!/usr/bin/python
# Partition the Yale face database into a training set and test set
# by removing a set of observations from each class.
import argparse
import os
import re
import shutil

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, help="path to image database", dest="PATH")
parser.add_argument("-s", "--samples", nargs="+", type=int, required=True, help="indices of samples to remove from each class", metavar="N", dest="SAMPLES")

args = parser.parse_args()

# initialize the training set and test set
TEMP_PATH = "temp"
TRAIN_PATH = "train_images"
TEST_PATH = "test_images"

if os.path.isdir(TEMP_PATH):
	os.rmdir(TEMP_PATH)

if os.path.isdir(TRAIN_PATH):
	shutil.rmtree(TRAIN_PATH)

if os.path.isdir(TEST_PATH):
	shutil.rmtree(TEST_PATH)

shutil.copytree(args.PATH, TEMP_PATH)

os.mkdir(TRAIN_PATH, 0775)
os.mkdir(TEST_PATH, 0755)

# partition the data set into training set and test set
for i in xrange(15):
	class_name = "subject%02d" % (i + 1)
	r = re.compile(class_name)
	class_files = sorted(filter(r.match, os.listdir(TEMP_PATH)))

	# move test images
	for i in args.SAMPLES:
		filename = class_files[i][(len(class_name) + 1):]
		src = os.path.join(TEMP_PATH, class_files[i])
		dst = os.path.join(TEST_PATH, "%s_%s" % (class_name, filename))
		shutil.move(src, dst);

	# move training images
	train_files = filter(r.match, os.listdir(TEMP_PATH))

	for f in train_files:
		filename = f[(len(class_name) + 1):]
		src = os.path.join(TEMP_PATH, f)
		dst = os.path.join(TRAIN_PATH, "%s_%s" % (class_name, filename))
		shutil.move(src, dst)

shutil.rmtree(TEMP_PATH)
