import cv2
import os
import shutil
import signal
import sys
import threading
import classifier
import detector

# define constants
DEVICE_NUM = 0

TRAIN_DATA = "../../train_data"
STREAM_DATA = "test_data"

MAX_FPS = 30

def display():
	# set the next timer call
	display.thread = threading.Timer(1.0 / MAX_FPS, display)
	display.thread.start()

	# check for reentrance
	if display.is_running:
		return

	display.is_running = True

	# capture frame
	ret, frame = cap.read()

	# detect faces
	faces = detector.detect_faces(frame, cascade_face)

	if len(faces) > 0:
		# prepare stream directory
		if os.path.exists(STREAM_DATA):
			shutil.rmtree(STREAM_DATA)

		if not os.path.exists(STREAM_DATA):
			os.mkdir(STREAM_DATA, 0755)

		# perform face recognition
		detector.crop_faces(frame, faces, STREAM_DATA)
		labels = server.predict(len(faces))

		print "Hello, %s" % (" and ".join(labels))

		# draw bounding boxes around each face
		frame = detector.label_faces(frame, faces, labels)

	# display the annotated frame
	cv2.imshow("Face Detection", frame)
	cv2.waitKey(1)

	display.is_running = False

display.thread = None
display.is_running = False

# load the cascade classifier files
print "Loading cascade classifier files..."

cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# initialize face-rec server
print "Initializing recognition server..."

server = classifier.Server()
server.train(TRAIN_DATA)
server.start(STREAM_DATA)

# initialize video stream
cap = cv2.VideoCapture(DEVICE_NUM)

# initialize display
display()

# initialize event loop
def quit(signal, frame):
	display.thread.cancel()
	cap.release()
	cv2.destroyAllWindows()
	server.stop()
	print "\nBye"

signal.signal(signal.SIGINT, quit)

