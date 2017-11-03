import os
import subprocess
import sys

FACE_REC = "../../face-rec"

class Server(object):
	def __init__(self):
		self._process = None

	def train(self, path):
		# check for face-rec executable
		if not os.path.isfile(FACE_REC):
			print "error: %s not found" % FACE_REC
			sys.exit(1)

		# run face-rec
		args = [FACE_REC, "--train", path, "--pca", "--pca_n1=20"]

		subprocess.Popen(args).wait()

	def start(self, path):
		# check for face-rec executable
		if not os.path.isfile(FACE_REC):
			print "error: %s not found" % FACE_REC
			sys.exit(1)

		# start face-rec stream
		args = [FACE_REC, "--stream", path, "--pca", "--pca_n1=20"]

		self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

	def predict(self, num_labels):
		# send prediction signal to face-rec
		self._process.stdin.write("1\n".encode())

		# read prediction data
		return [self._process.stdout.readline().replace("\n", "").split()[1] for i in xrange(num_labels)]

	def stop(self):
		self._process.communicate("0\n");

