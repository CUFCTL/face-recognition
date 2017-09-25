#!/usr/bin/python
# Detect all faces in an image and save each face as a cropped image.
import cv2
from datetime import datetime

def detect_face(img, cascade_face):
	# read the input image and convert to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# run the detection function on the grayscale image
	faces = cascade_face.detectMultiScale(img_gray, 1.3, 5)

	return faces

def box_face(img, faces):
	for(x,y,w,h) in faces:
		x1 = x
		x2 = x + w
		y1 = y
		y2 = y + h

		cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

	return img

def crop_face(img, faces, FOLDER_OUT):
	# save each face to a cropped image, and draw rectangles
	for (x,y,w,h) in faces:
		path = FOLDER_OUT + "/" + str(datetime.now()) + ".ppm"
		print path

		x1 = x
		x2 = x + w
		y1 = y
		y2 = y + h

		img_out = img[y1:y2, x1:x2]
		img_out = cv2.resize(img_out, (128,128))
		cv2.imwrite(path, img_out)
