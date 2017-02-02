#!/usr/bin/python
# Detect all faces in an image and save each face as a cropped image.
import os
import subprocess
import sys
import time
import cv2

# parse command-line arguments
FNAME_IN = ""
BOUNDING_BOX = False

for i in xrange(1, len(sys.argv)):
    arg = sys.argv[i]

    if arg == "-b" or arg == "--bounding-box":
        BOUNDING_BOX = True
    else:
        FNAME_IN = arg

if len(FNAME_IN) == 0:
    print "usage: python crop.py [options] [input-image]"
    print "options:"
    print "  -b, --bounding-box  save original image with bounding boxes"
    sys.exit(1)

parts = os.path.splitext(FNAME_IN)
FOLDER_OUT = parts[0] + "_cropped"
FNAME_OUT = FOLDER_OUT + parts[1]

# load the cascade classifier files
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# read the input image and convert to grayscale
img = cv2.imread(FNAME_IN)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run the detection function on the grayscale image
faces = cascade_face.detectMultiScale(img_gray, 1.3, 5)

# create directory for cropped images
if not os.path.isdir(FOLDER_OUT):
    os.mkdir(FOLDER_OUT, 0775)

# save each face to a cropped image, and draw rectangles
i = 0
for (x,y,w,h) in faces:
    path = FOLDER_OUT + "/" + str(i) + ".jpg"
    print path

    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h

    roi = img[y1:y2, x1:x2]
    cv2.imwrite(path, roi)

    if BOUNDING_BOX:
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

    i = i + 1

# save image with rectangles
if BOUNDING_BOX:
    cv2.imwrite(FNAME_OUT, img)
