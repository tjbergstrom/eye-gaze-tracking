# eyez.py
# August 2020
# First attempts at extracting eyes from an image
#
# python3 eyez.py


from scipy.spatial import distance as dist
from collections import OrderedDict
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

def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def img_show(txt, img):
	cv2.imshow(txt, img)
	cv2.waitKey(0)

eye_landmarks = OrderedDict([
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices of the facial landmarks for the left and right eyes
(lefteye_start, lefteye_end) = eye_landmarks["left_eye"]
(righteye_start, righteye_end) = eye_landmarks["right_eye"]
#(lefteye_start, lefteye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(righteye_start, righteye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

eyes_closed = False
closed_ratio = .3

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
	shape = shape_to_np(shape)
	#shape = face_utils.shape_to_np(shape)
	# Left and right eye coordinates
	left_eye = shape[lefteye_start:lefteye_end]
	right_eye = shape[righteye_start:righteye_end]
	# Aspect ratios
	left_aspect_ratio = eye_aspect_ratio(left_eye)
	right_aspect_ratio = eye_aspect_ratio(right_eye)
	eyez_aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.
	if eyez_aspect_ratio < closed_ratio:
		print("detected eyes closed or blinking")
		eyes_closed = True
		#continue
	# Extracting eyes
	left_eye_hull = cv2.convexHull(left_eye)
	right_eye_hull = cv2.convexHull(right_eye)
	extracted_eyes.append(cv2.boundingRect(left_eye_hull))
	extracted_eyes.append(cv2.boundingRect(right_eye_hull))
	cv2.drawContours(img, [left_eye_hull], -1, (255, 255, 0), 2)
	cv2.drawContours(img, [right_eye_hull], -1, (255, 255, 0), 2)

img_show(img_path, img)

if not eyes_closed:
	for eye in extracted_eyes:
		(x, y, w, h) = eye
		eye = img[y-2:y+h+2, x-2:x+w+2]
		img_show("eye", eye)



#
