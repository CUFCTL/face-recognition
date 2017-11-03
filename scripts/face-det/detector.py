import cv2
import datetime

def detect_faces(image, cascade_face):
	# read the input image and convert to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# run the detection function on the grayscale image
	faces = cascade_face.detectMultiScale(image, 1.3, 5)

	return faces

def label_faces(image, faces, labels):
	# draw bounding box and label for each face
	for i in xrange(len(faces)):
		(x,y,w,h) = faces[i]
		x1 = x
		x2 = x + w
		y1 = y
		y2 = y + h

		cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
		cv2.putText(image, labels[i], (x1,y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)

	return image

def crop_faces(image, faces, path):
	# crop and save each face to a file
	for (x,y,w,h) in faces:
		path = "%s/%s.ppm" % (path, str(datetime.datetime.now()))
		path = path.replace(" ", "_")

		x1 = x
		x2 = x + w
		y1 = y
		y2 = y + h

		image_out = image[y1:y2, x1:x2]
		image_out = cv2.resize(image_out, (128,128))
		cv2.imwrite(path, image_out)
