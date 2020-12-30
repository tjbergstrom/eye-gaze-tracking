# trackeye.py
# $ python3 trackeye.py
# Dec. 2020
# Starting over,
# Maybe just try to detect eye colors, that seems do-able right?
# And maybe later try to detect gaze direction


from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2


class Eye:
	def __init__(self, LR=""):
		self.eye = None
		self.box = None
		self.img = None
		self.hull = None
		self.color = ""
		self.RGB = (0,0,0)
		self.closed = False
		self.LR = LR

	def __repr__(self):
		return f"{self.LR}: {self.box}"


def aspect_ratio(eye):
	# Euclidean distances between vertical xy coords
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Euclidean distance between horizontal xy coords
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2. * C)


def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def extract_eyes(predictor, detector, img):
	eyes = []
	faces = detector(gray, 0)
	for face in faces:
		left, right = Eye("L"), Eye("R")
		shape = shape_to_np(predictor(img, face))
		left.eye = shape[42 : 48]
		right.eye = shape[36 : 42]
		if aspect_ratio(left.eye) < 0.2:
			left.closed = True
		if aspect_ratio(right.eye) < 0.2:
			right.closed = True
		left.hull = cv2.convexHull(left.eye)
		right.hull = cv2.convexHull(right.eye)
		left.box = cv2.boundingRect(left.hull)
		right.box = cv2.boundingRect(right.hull)
		eyes.append(left)
		eyes.append(right)
	return eyes


def eye_colors(eyes, img):
	for eye in eyes:
		if eye.closed:
			continue
		(x, y, w, h) = eye.box
		eye.img = img[y-2 : y+h+2, x-2 : x+w+2]
		cv2.imshow("", eye.img)
		cv2.waitKey(10)




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str, default="images/photo_1.jpg")
	args = vars(ap.parse_args())

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	img_path = args["image"]
	img = cv2.imread(img_path)
	img = imutils.resize(img, 500)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	eyes = extract_eyes(predictor, detector, gray)

	eye_colors(eyes, img)

	print(eyes)



##
