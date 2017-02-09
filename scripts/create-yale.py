#!/usr/bin/python
# Partition the Yale face database into a training set and test set
# by removing a set of observations from each class.
import argparse
import os
import re
import shutil
import subprocess

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, help="path to image database", dest="PATH")
parser.add_argument("-s", "--samples", nargs="+", type=int, required=True, help="indices of samples to remove from each class", metavar="N", dest="SAMPLES")

args = parser.parse_args()

# initialize the training set and test set
TEMP_PATH1 = "temp1"
TEMP_PATH2 = "temp2"
TRAIN_PATH = "train_images"
TEST_PATH = "test_images"

if os.path.isdir(TEMP_PATH1):
	os.rmdir(TEMP_PATH1)

if os.path.isdir(TEMP_PATH2):
	os.rmdir(TEMP_PATH2)

if os.path.isdir(TRAIN_PATH):
	shutil.rmtree(TRAIN_PATH)

if os.path.isdir(TEST_PATH):
	shutil.rmtree(TEST_PATH)

shutil.copytree(args.PATH, TEMP_PATH1)
os.remove(os.path.join(TEMP_PATH1, "Readme.txt"))

# convert images from GIF to PGM
files = os.listdir(TEMP_PATH1)
for f in files:
	src = os.path.join(TEMP_PATH1, f)
	dst = os.path.join(TEMP_PATH1, f + ".gif")
	shutil.move(src, dst)

devnull = open(os.devnull, "w")
subprocess.call(["./scripts/convert-images.sh", TEMP_PATH1, TEMP_PATH2, "gif", "pgm"], stdout=devnull)

os.mkdir(TRAIN_PATH, 0775)
os.mkdir(TEST_PATH, 0755)

# partition the data set into training set and test set
for i in xrange(15):
	class_name = "subject%02d" % (i + 1)
	r = re.compile(class_name)
	class_files = sorted(filter(r.match, os.listdir(TEMP_PATH2)))

	# move test images
	for i in args.SAMPLES:
		filename = class_files[i][(len(class_name) + 1):]
		src = os.path.join(TEMP_PATH2, class_files[i])
		dst = os.path.join(TEST_PATH, "%s_%s" % (class_name, filename))
		shutil.move(src, dst);

	# move training images
	train_files = filter(r.match, os.listdir(TEMP_PATH2))

	for f in train_files:
		filename = f[(len(class_name) + 1):]
		src = os.path.join(TEMP_PATH2, f)
		dst = os.path.join(TRAIN_PATH, "%s_%s" % (class_name, filename))
		shutil.move(src, dst)

shutil.rmtree(TEMP_PATH1)
shutil.rmtree(TEMP_PATH2)
