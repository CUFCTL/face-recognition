#!/usr/bin/python
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the face recognition system with a dataset. The dataset should
# have a script of the form "./scripts/create-[dataset].py" that
# can partition the dataset into training and test sets.

import os
import shutil
import subprocess
import sys


def alg():
	# check for face-rec executable
	if not os.path.isfile("./face-rec"):
		print "error: ./face-rec not found"
		sys.exit(1)

	args_c = ["./face-rec", "--test", "test_data"]

	# construct results object
	results = []

	# perform repeated random testing
	for i in xrange(10):

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

	print "".join(["{0:12.3f}".format(r) for r in results])
