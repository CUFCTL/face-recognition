#!/usr/bin/python
# Download and extract the MNIST dataset.
import os
import shutil
import struct
import subprocess
import sys

def read_int(fp):
	return struct.unpack(">i", fp.read(4))[0]

def read_labels(fname):
	fp = open(fname, "rb")

	# read magic number
	magic_number = read_int(fp)
	if magic_number != 0x0801:
		print "error: cannot read %s" % (fname)
		sys.exit(1)

	# read array of labels
	num_labels = read_int(fp)
	labels_str = fp.read(num_labels)
	labels = [ord(c) for c in labels_str]

	fp.close();

	return labels

def read_images(fname):
	fp = open(fname, "rb")

	# read magic number
	magic_number = read_int(fp)
	if magic_number != 0x0803:
		print "error: cannot read %s" % (fname)
		sys.exit(1)

	# read array of images
	num_images = read_int(fp)
	rows = read_int(fp)
	cols = read_int(fp)
	images = [fp.read(rows * cols) for i in xrange(num_images)]

	fp.close()

	return images

if not os.path.exists("datasets"):
	os.mkdir("datasets", 0775)
os.chdir("datasets")

files = [
	"train-images-idx3-ubyte",
	"train-labels-idx1-ubyte",
	"t10k-images-idx3-ubyte",
	"t10k-labels-idx1-ubyte"
]

# download archives if necessary
for f in files:
	archive = f + ".gz"
	if not os.path.exists(archive):
		subprocess.call("wget http://yann.lecun.com/exdb/mnist/%s" % (archive), shell=True)

# extract archives
for f in files:
	subprocess.call("gzip -d %s.gz" % (f), shell=True)

# transform IDX files into directory tree
DB_PATH = "mnist"

if os.path.isdir(DB_PATH):
	shutil.rmtree(DB_PATH)

os.mkdir(DB_PATH)
for i in xrange(10):
	os.mkdir("%s/%d" % (DB_PATH, i))

counts = [1 for i in xrange(10)]
for dname in ["t10k"]:
	labels = read_labels("%s-labels-idx1-ubyte" % (dname))
	images = read_images("%s-images-idx3-ubyte" % (dname))

	if len(labels) != len(images):
		print "error: %s labels and images do not match" % (dname)
		sys.exit(1)

	for i in xrange(len(labels)):
		label = labels[i]
		image = images[i]

		fp = open("%s/%d/%04d.pgm" % (DB_PATH, label, counts[label]), "wb")
		fp.write("P5\n")
		fp.write("%d %d %d\n" % (28, 28, 255))
		fp.write(image)
		fp.close()

		counts[label] += 1

# cleanup
for f in files:
	os.remove(f)
