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
import statistics
from imutils import build_montages


class Eye:
	def __init__(self, LR=""):
		self.points = []
		self.box = (0,0,0,0)
		self.img = None
		self.hull = None
		self.iris = None
		self.pupil = (0,0)
		self.color = ""
		self.BGR = (0,0,0)
		self.closed = False
		self.LR = LR
		self.center_xy = (0,0)

	def __repr__(self):
		return f"{self.LR}: {self.pupil}, {self.BGR}"


def aspect_ratio(eye):
	# Vertical xy coords
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal xy coords
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
		left.hull = cv2.convexHull(left.points)
		right.hull = cv2.convexHull(right.points)
		left.box = cv2.boundingRect(left.hull)
		right.box = cv2.boundingRect(right.hull)
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
		thresh = cv2.erode(thresh, None, iterations=2)
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
			return
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
			return
		b = int(b // len(bgr))
		g = int(g // len(bgr))
		r = int(r // len(bgr))
		eye.BGR = (b,g,r)


def sample_output(eyes, img, sz=200):
	imgs = [img]
	img_cpy = img.copy()
	for eye in eyes:
		if eye.closed:
			continue
		imgs.append(eye.iris)
		color = np.zeros((sz, sz, 3), np.uint8)
		color[:,:] = eye.BGR
		imgs.append(color)
		cv2.circle(img_cpy, eye.pupil, 13, (255,255,255), -1)
	imgs.append(img_cpy)
	imgs_cpy = [imgs[0], imgs[1], imgs[3], imgs[5], imgs[2], imgs[4]]
	m = build_montages(imgs_cpy, (sz, sz), (3, 2))[0]
	cv2.imshow("", m)
	cv2.waitKey(0)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str, default="images/photo_9.jpg")
	args = vars(ap.parse_args())

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	img_path = args["image"]
	img = cv2.imread(img_path)
	img = imutils.resize(img, 500)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	eyes = extract_eyes(predictor, detector, gray)

	find_centers(eyes, img)
	find_colors(eyes)

	sample_output(eyes, img)

	print(eyes)



##
