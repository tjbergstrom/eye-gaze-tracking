# eyez.py
# August 2020
# First attempts at extracting eyes from an image


from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def resize_preserve_aspect_ratio(img, width):
    if width is None or width < 1:
        return img
    (h, w) = img.shape[:2]
    ratio = width / float(w)
    dims = (width, int(h * ratio))
    img = cv2.resize(img, dims)
    return img

def eye_aspect_ratio(eye):
	# Euclidean distances between vertical eye landmarks x,y coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Euclidean distance between horizontal eye landmark x,y coordinates
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2. * C)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices of the facial landmarks for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

blink_ratio = .3
blink = False

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="images/photo_1.jpg")
args = vars(ap.parse_args())
img_path = args["image"]
img = cv2.imread(img_path)
img = resize_preserve_aspect_ratio(img, 500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 0)
extracted_eyes = []

for face in faces:
	# Facial landmarks
	shape = predictor(gray, face)
	shape = face_utils.shape_to_np(shape)
	# Left and right eye coordinates
	left_eye = shape[lStart:lEnd]
	right_eye = shape[rStart:rEnd]
	# Aspect ratios
	left_aspect_ratio = eye_aspect_ratio(left_eye)
	right_aspect_ratio = eye_aspect_ratio(right_eye)
	ear = (left_aspect_ratio + right_aspect_ratio) / 2.
	if ear < blink_ratio:
		blink = True
	# Extracting eyes
	left_eye_hull = cv2.convexHull(left_eye)
	right_eye_hull = cv2.convexHull(right_eye)
	extracted_eyes.append(cv2.boundingRect(left_eye_hull))
	extracted_eyes.append(cv2.boundingRect(right_eye_hull))
	#cv2.drawContours(img, [left_eye_hull], -1, (255, 255, 0), 2)
	#cv2.drawContours(img, [right_eye_hull], -1, (255, 255, 0), 2)

cv2.imshow(img_path, img)
cv2.waitKey(0)

for box in extracted_eyes:
	(x, y, w, h) = box
	#cv2.rectangle(img, (x,y), (w,h), (0,0,255), 2)
	box = img[y-2:y+h+2, x-2:x+w+2]
	cv2.imshow(img_path, box)
	cv2.waitKey(0)



#
