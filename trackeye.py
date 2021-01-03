# trackeye.py
# $ python3 trackeye.py
# Dec. 2020
#
# Using the Dlib facial landmarks points detector to extract eyes.
# And figuring out some tricks to actually locate the iris and pupil.
# And caclulate the iris color.


from scipy.spatial import distance as dist
import numpy as np
import statistics
import imutils
import dlib
import cv2


class Eye:
	def __init__(self, LR=""):
		self.LR = LR
		self.points = []
		self.box = (0,0,0,0)
		self.center_xy = (0,0)
		self.pupil = (0,0)
		self.img = None
		self.iris = None
		self.BGR = (0,0,0)
		self.closed = False

	def __repr__(self):
		return f"{self.LR}: {self.pupil}"


def aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)


def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def extract_eyes(predictor, detector, gray):
	eyes = []
	faces = detector(gray, 0)
	for face in faces:
		left, right = Eye("L"), Eye("R")
		shape = shape_to_np(predictor(gray, face))
		left.points = shape[42 : 48]
		right.points = shape[36 : 42]
		if aspect_ratio(left.points) < 0.2:
			left.closed = True
		if aspect_ratio(right.points) < 0.2:
			right.closed = True
		left_hull = cv2.convexHull(left.points)
		right_hull = cv2.convexHull(right.points)
		left.box = cv2.boundingRect(left_hull)
		right.box = cv2.boundingRect(right_hull)
		eyes.append(left)
		eyes.append(right)
	return eyes


def find_centers(eyes, img):
	for eye in eyes:
		if eye.closed:
			continue
		(x, y, w, h) = eye.box
		eye.img = img[y : y+h, x : x+w]
		eye_img = cv2.cvtColor(eye.img, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(eye_img, 125, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		thresh = cv2.bitwise_not(thresh)
		contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		if not contours:
			continue
		contour = max(contours, key=cv2.contourArea)
		moment = cv2.moments(contour)
		cx = int(moment["m10"] // moment["m00"])
		cy = int(moment["m01"] // moment["m00"])
		d = eye.img.shape[0]
		r = d // 2
		if cx - r < 0:
			cx = r
		eye.iris = eye.img[0 : d, cx-r : cx+r]
		eye.pupil = (x+cx, y+cy)
		eye.center_xy = (x+w//2, y+h//2)


def find_colors(eyes):
	for eye in eyes:
		if eye.closed:
			continue
		if eye.iris.shape[0] < 10:
			continue
		bgr = []
		mid_x = eye.iris.shape[1] // 2
		qtr_x = eye.iris.shape[1] // 8
		for x in range(mid_x-qtr_x+1, mid_x+qtr_x):
			y = eye.iris.shape[0] - 4
			prev_avg = None
			while y > eye.iris.shape[0] // 2:
				y -= 1
				avg = statistics.mean(eye.iris[y,x])
				if prev_avg is None:
					prev_avg = avg
					continue
				prev_avg = avg
				if abs(avg - prev_avg) > 24:
					continue
				if abs(avg - prev_avg) > 36:
					break
				bgr.append(eye.iris[y,x])
		(b,g,r) = (0,0,0)
		for i in bgr:
			b += i[0]
			g += i[1]
			r += i[2]
		if len(bgr) == 0:
			continue
		b = int(b // len(bgr))
		g = int(g // len(bgr))
		r = int(r // len(bgr))
		eye.BGR = (b,g,r)



##
