import cv2
import os
import shutil
import subprocess
import sys
import crop

# define constants
DEVICE_NUM = 0
NUM_FRAMES = 5

FACE_REC = "../../face-rec"
TRAIN_DATA = "../../train_data"
STREAM_DATA = "test_data"

# load the cascade classifier files
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

def train():
	# check for face-rec executable
	if not os.path.isfile(FACE_REC):
		print "error: %s not found" % FACE_REC
		sys.exit(1)

	# run face-rec
	args = [FACE_REC, "--train", TRAIN_DATA, "--pca", "--pca_n1=50"]

	p = subprocess.Popen(args, stdout=subprocess.PIPE)
	p.wait()

def stream_init():
	# check for face-rec executable
	if not os.path.isfile(FACE_REC):
		print "error: %s not found" % FACE_REC
		sys.exit(1)

	# start face-rec stream
	args = [FACE_REC, "--stream", STREAM_DATA, "--pca", "--pca_n1=50"]

	return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def stream_predict(process, num_labels):
	# send prediction signal to face-rec
	process.stdin.write("1\n".encode())

	# read prediction data
	labels = [process.stdout.readline().replace("\n", "").split()[1] for i in xrange(num_labels)]

	print "Hello, %s" % (" and ".join(labels))

	return labels

def stream_finalize(process):
	process.communicate("0\n");

# train a model on the dataset
train()

# initialize face-rec stream
process = stream_init()

# initialize video stream
cap = cv2.VideoCapture(DEVICE_NUM)

# begin capture/detect/recognize loop
i = 0

while True:
	# capture frame
	ret, frame = cap.read()

	# detect faces
	faces = crop.detect_face(frame, cascade_face)

	# perform recognition every NUM_FRAMES
	i = (i + 1) % NUM_FRAMES

	if i == 0 and len(faces) > 0:
		# prepare stream directory
		if os.path.exists(STREAM_DATA):
			shutil.rmtree(STREAM_DATA)

		if not os.path.exists(STREAM_DATA):
			os.mkdir(STREAM_DATA, 0755)

		# perform face recognition
		crop.crop_face(frame, faces, STREAM_DATA)
		labels = stream_predict(process, len(faces))
	else:
		labels = []

	# draw bounding boxes around each face
	img = crop.box_face(frame, faces, labels)

	# display the annotated frame
	cv2.imshow("Face Detection", img)

	if (cv2.waitKey(1) & 0xFF) == ord("q"):
		break

# cleanup
stream_finalize(process)
cap.release()
cv2.destroyAllWindows()
