# eyez.py
# August 2020
# Extracting eyes from an image
#
# python3 eyez.py
#
# todo:
# detect eyes X
# extract eyes X
# detect eyes closed X
# extract the iris and pupil
# detect gaze direction
# draw gaze direction
# detect eye color ?


from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
from collections import OrderedDict
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

eye_color_palette = [
	([69, 24, 0], [99, 72, 30]), # brown
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

# Indices of the facial landmarks for left and right eyes
eye_landmarks = OrderedDict([
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
])
(lefteye_start, lefteye_end) = eye_landmarks["left_eye"]
(righteye_start, righteye_end) = eye_landmarks["right_eye"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

extracted_eyes = []
eyes_closed = blinking = False
left_closed = right_closed = False
closed_ratio = .2
blink_ratio = .3

info = []
display_info = True

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="images/photo_1.jpg")
args = vars(ap.parse_args())
img_path = args["image"]
img = cv2.imread(img_path)
img = resize_preserve_aspect_ratio(img, 500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 0)

for face in faces:
	# Facial landmarks
	shape = predictor(gray, face)
	shape = shape_to_np(shape)
	# Left and right eye coordinates
	left_eye = shape[lefteye_start:lefteye_end]
	right_eye = shape[righteye_start:righteye_end]
	# Aspect ratios
	left_aspect_ratio = eye_aspect_ratio(left_eye)
	if left_aspect_ratio < closed_ratio:
		info.append("detected LEFT eye closed or winking")
		left_closed = True
	right_aspect_ratio = eye_aspect_ratio(right_eye)
	if right_aspect_ratio < closed_ratio:
		info.append("detected RIGHT eye closed or winking")
		right_closed = True
	eyez_aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.
	if eyez_aspect_ratio < blink_ratio:
		info.append("detected blinking")
		blinking = True
	if left_closed and right_closed:
		info.append("detected both eyes closed")
		eyes_closed = True
	# Extracting eyes
	left_eye_hull = cv2.convexHull(left_eye)
	right_eye_hull = cv2.convexHull(right_eye)
	extracted_eyes.append((cv2.boundingRect(left_eye_hull), left_closed))
	extracted_eyes.append((cv2.boundingRect(right_eye_hull), right_closed))
	#cv2.drawContours(img, [left_eye_hull], -1, (255, 255, 0), 2)
	#cv2.drawContours(img, [right_eye_hull], -1, (255, 255, 0), 2)

img_show(img_path, img)

def RGB2HEX(color):
	R = int(color[0])
	G = int(color[1])
	B = int(color[2])
	return "#{:02x}{:02x}{:02x}".format(R, G, B)

def eye_color(eye):
	eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
	eye = eye.reshape(eye.shape[0] * eye.shape[1], 3)
	clf = KMeans(n_clusters=6)
	labels = clf.fit_predict(eye)
	counts = Counter(labels)
	center_colors = clf.cluster_centers_
	ordered_colors = [center_colors[i] for i in counts.keys()]
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]
	print("\nhex_colors", hex_colors)
	print("\nlabels", labels)
	print("\nordered_colors", ordered_colors)

if not eyes_closed:
	for (eye, closed) in extracted_eyes:
		if not closed:
			(x, y, w, h) = eye
			eye = img[y-2:y+h+2, x-2:x+w+2]
			img_show("eye", eye)
			eye_color(eye)
			gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
			thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
			#thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			#img_show("eye", thresh)
			#thresh = cv2.erode(thresh, None, iterations=2)
			#img_show("eye", thresh)
			#thresh = cv2.dilate(thresh, None, iterations=4)
			#img_show("eye", thresh)
			#thresh = cv2.medianBlur(thresh, 3)
			#img_show("eye", thresh)

if display_info:
	for msg in info:
		print(msg)



#
