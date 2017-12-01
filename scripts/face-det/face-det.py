import cv2
import datetime
import os
import shutil
import signal
import sys
import threading

# define constants
DEVICE_NUM = 0
OUTPUT_DIR = "test_data"
MAX_FPS = 30

IMAGE_SIZE = (128, 128)
RECT_COLOR = (255, 0, 0)
RECT_LINEWIDTH = 2

def detect_faces(image, cascade_face):
	# read the input image and convert to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# run the detection function on the grayscale image
	faces = cascade_face.detectMultiScale(image, 1.3, 5)

	return faces

def label_faces(image, faces):
	# draw bounding box for each face
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w,y+h), RECT_COLOR, RECT_LINEWIDTH)

	return image

def crop_faces(image, faces, path):
	# crop and save each face to a file
	for (x,y,w,h) in faces:
		filename = "%s/%s.ppm" % (path, str(datetime.datetime.now()))
		filename = filename.replace(" ", "_")

		print "Saving '%s'..." % filename

		image_out = image[y:(y+h), x:(x+w)]
		image_out = cv2.resize(image_out, IMAGE_SIZE)
		cv2.imwrite(filename, image_out)

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

	if not ret:
		print "error: could not read frame from video stream"
		quit()

	# detect faces
	faces = detect_faces(frame, cascade_face)

	if len(faces) > 0:
		# save each face to a file
		crop_faces(frame, faces, OUTPUT_DIR)

		# draw bounding boxes around each face
		frame = label_faces(frame, faces)

	# display the annotated frame
	cv2.imshow("Face Detection", frame)
	cv2.waitKey(1)

	display.is_running = False

display.thread = None
display.is_running = False

def quit(signal=None, frame=None):
	if display.thread:
		display.thread.cancel()
	cap.release()
	cv2.destroyAllWindows()
	sys.exit()
	print "\nBye"

# load the cascade classifier files
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# prepare output directory
if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR, 0755)

# initialize video stream
cap = cv2.VideoCapture(DEVICE_NUM)

if not cap.isOpened():
	print "error: could not open video stream"
	quit()

# initialize display
display()

# initialize event loop
signal.signal(signal.SIGINT, quit)
