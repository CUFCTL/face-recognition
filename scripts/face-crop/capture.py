import numpy as np
import cv2
import os
import crop

# define constants
FOLDER_OUT = "FOLDER_OUT"
DEVICE_NUM = 0
NUM_FRAMES = 5

# load the cascade classifier files
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# initialize output directory
if not os.path.exists(FOLDER_OUT):
	os.mkdir(FOLDER_OUT, 0775)

# initialize video feed
cap = cv2.VideoCapture(DEVICE_NUM)

# begin capture/crop loop
i = 0
while True:
	# capture frame
	ret, frame = cap.read()

	# detect and draw bounding box for each face
	faces = crop.detect_face(frame, cascade_face)
	img = crop.box_face(frame, faces)

	# save cropped image every NUM_FRAMES
	i = i + 1
	if i == NUM_FRAMES:
		crop.crop_face(frame, faces, FOLDER_OUT)
		i = 0

	# display the annotated frame
	cv2.imshow("Face Detection", img)

	if (cv2.waitKey(1) & 0xFF) == ord("q"):
		break

# release the video feed
cap.release()
cv2.destroyAllWindows()
