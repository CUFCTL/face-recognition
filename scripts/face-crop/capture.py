import numpy as np
import cv2
import os
from crop import detect_face
from crop import box_face
from crop import crop_face

# load the cascade classifier files
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#cascade_eye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

FOLDER_OUT = "FOLDER_OUT"
os.mkdir(FOLDER_OUT, 0775)


cap = cv2.VideoCapture(0)

i = 0
while(True):
    # Capture frame-by-frame
    #for i in range(0,9):
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face(frame, cascade_face)
    img = box_face(frame, faces)
    if i == 9:
        crop_face(frame, faces, FOLDER_OUT)
        i = -1

    i = i + 1
    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
