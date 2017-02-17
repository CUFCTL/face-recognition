#!/usr/bin/python
# Create a training set and test set from a dataset
# by removing a set of observations from each class.
import argparse
import os
import re
import shutil
import sys

class Dataset(object):
	def __init__(self, path, num_classes, class_size):
		self.PATH = path
		self.NUM_CLASSES = num_classes
		self.CLASS_SIZE = class_size

	def get_class_name(self, i):
		raise NotImplementedError("Every Dataset must implement the get_class_name method!")

	def get_class_path(self, path, i):
		raise NotImplementedError("Every Dataset must implement the get_class_path method!")

	def get_class_files(self, path, i):
		raise NotImplementedError("Every Dataset must implement the get_class_files method!")

	def get_dst_filename(self, i, filename):
		return filename

class MNISTDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/mnist", 10, 800)

	def get_class_name(self, i):
		return str(i)

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return sorted(os.listdir(class_path))

class ORLDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/orl_faces", 40, 10)

	def get_class_name(self, i):
		return "s%d" % (i + 1)

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return sorted(os.listdir(class_path))

class YaleDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/yalefaces", 15, 11)

	def get_class_name(self, i):
		return "subject%02d" % (i + 1)

	def get_class_path(self, path, i):
		return path

	def get_class_files(self, path, i):
		class_regex = re.compile(self.get_class_name(i))
		return sorted(filter(class_regex.match, os.listdir(path)))

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return filename[(len(class_name) + 1):]

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", choices=["mnist", "orl", "yale"], required=True, help="name of dataset", dest="DATASET")
parser.add_argument("-t", "--train", nargs="+", type=int, help="indices of samples for training set", metavar="N", dest="TRAIN_SAMPLES")
parser.add_argument("-r", "--test", nargs="+", type=int, help="indices of samples for test set", metavar="N", dest="TEST_SAMPLES")

args = parser.parse_args()

# determine parameters of dataset
if args.DATASET == "mnist":
	dataset = MNISTDataset()
elif args.DATASET == "orl":
	dataset = ORLDataset()
elif args.DATASET == "yale":
	dataset = YaleDataset()

# determine sample sets
if args.TRAIN_SAMPLES is None and args.TEST_SAMPLES is None:
	print "error: you must specify training samples and/or test samples"
	sys.exit(1)
elif args.TRAIN_SAMPLES is None:
	args.TRAIN_SAMPLES = list(set(range(dataset.CLASS_SIZE)) - set(args.TEST_SAMPLES))
elif args.TEST_SAMPLES is None:
	args.TEST_SAMPLES = list(set(range(dataset.CLASS_SIZE)) - set(args.TRAIN_SAMPLES))

# verify that sample sets are disjoint
if not set(args.TRAIN_SAMPLES).isdisjoint(set(args.TEST_SAMPLES)):
	print "error: training set and test set must be disjoint"
	sys.exit(1)

# initialize the training set and test set
TRAIN_PATH = "train_images"
TEST_PATH = "test_images"

if os.path.isdir(TRAIN_PATH):
	shutil.rmtree(TRAIN_PATH)

if os.path.isdir(TEST_PATH):
	shutil.rmtree(TEST_PATH)

os.mkdir(TRAIN_PATH, 0775)
os.mkdir(TEST_PATH, 0775)

# partition the data set into training set and test set
for i in xrange(dataset.NUM_CLASSES):
	class_name = dataset.get_class_name(i)
	class_path = dataset.get_class_path(dataset.PATH, i)
	class_files = dataset.get_class_files(dataset.PATH, i)

	# move training images
	train_files = [class_files[i] for i in args.TRAIN_SAMPLES]

	for f in train_files:
		filename = dataset.get_dst_filename(i, f)
		src = os.path.join(class_path, f)
		dst = os.path.join(TRAIN_PATH, "%s_%s" % (class_name, filename))
		shutil.copy(src, dst)

	# move test images
	test_files = [class_files[i] for i in args.TEST_SAMPLES]

	for f in test_files:
		filename = dataset.get_dst_filename(i, f)
		src = os.path.join(class_path, f)
		dst = os.path.join(TEST_PATH, "%s_%s" % (class_name, filename))
		shutil.copy(src, dst);
