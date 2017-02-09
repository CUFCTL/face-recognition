#!/usr/bin/python
# Partition the ORL face database into a training set and test set
# by removing a set of observations from each class.
import argparse
import os
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

shutil.rmtree(TRAIN_PATH)
shutil.rmtree(TEST_PATH)

shutil.copytree(args.PATH, TEMP_PATH)

os.mkdir(TRAIN_PATH, 0775)
os.mkdir(TEST_PATH, 0755)

# partition the data set into training set and test set
for i in xrange(40):
	class_name = "s%d" % (i + 1)
	class_path = os.path.join(TEMP_PATH, class_name)
	class_files = sorted(os.listdir(class_path))

	# move test images
	for i in args.SAMPLES:
		src = os.path.join(class_path, class_files[i])
		dst = os.path.join(TEST_PATH, "%s_%s" % (class_name, class_files[i]))
		shutil.move(src, dst);

	# move training images
	train_files = os.listdir(class_path)

	for f in train_files:
		src = os.path.join(class_path, f)
		dst = os.path.join(TRAIN_PATH, "%s_%s" % (class_name, f))
		shutil.move(src, dst)

shutil.rmtree(TEMP_PATH)
