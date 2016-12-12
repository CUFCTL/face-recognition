#!/usr/bin/python
# This script will take an input directory and crop all the faces in the
# subdirectories. Ideal use would be for orl_faces

import sys
import os
import subprocess
import numpy as np
import cv2
from time import sleep

# function to split up the path of an image into a list
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# check for proper input arguments
if len(sys.argv) != 3:
    print '\n  ***USAGE: python crop.py in_img.jpg out_img.jpg***\n'
    sys.exit(1)

# split up the path and return the filename without an extension
file_comps = splitall(sys.argv[2])
filename = os.path.splitext(file_comps[-1])[0]

# load the cascade classifier files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# read in the input image and convert to grayscale
img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run the detection function on the gray image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

i = 0

# draw a rectangle around each face that is found, then save it to a new file
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y-15),(x+w,y+h+15),(255,0,0),2)
    roi = img[y-15:y+h+15, x:x+w]
    path = './cropped/' + filename + '_' + str(i) + '.jpg'
    print(path)
    cv2.imwrite('./cropped/' + filename + '_' + str(i) + '.jpg', roi)
    i = i + 1

cv2.imshow('img',img)
cv2.imwrite(sys.argv[2], img)
cv2.waitKey(0)
cv2.destroyAllWindows()
