# gaze_direction.py
# $ python3 gaze_direction.py
# Jan 1 2021
#
# With an input a video file...
# Get the gaze direction, assuming only one person in the frame.
# Calculate the face tilt direction and the eye gaze direction,
# and use these to determine relatively which direction the person is looking.
# Assuming the person is at the center of the frame.


import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import trackeye
import imutils
import math


class Face_Tilt:
    def __init__(self):
        #   Five tilts:
        #     1
        #   3 4 5
        #     7
        self.x_tilt = 4 # right=3 forward=4 left=5
        self.y_tilt = 4 # up=1 forward=4 down=7
        self.upr_lip = 0
        self.chin = 0

    def determine_x_tilt(self, shape):
        right = dist.euclidean(shape[30], shape[31])
        left = dist.euclidean(shape[30], shape[35])
        if right > left + left * 0.80:
            self.x_tilt = 5
        elif left > right + right * 0.80:
            self.x_tilt = 3
        else:
            self.x_tilt = 4

    def determine_y_tilt(self, shape):
        upr_lip = dist.euclidean(shape[33], shape[51])
        chin = dist.euclidean(shape[8], shape[57])
        if upr_lip > self.upr_lip:
            self.upr_lip = upr_lip
        if chin > self.chin:
            self.chin = chin
        if upr_lip < self.upr_lip * 0.70 and chin < self.chin * 0.70:
            self.y_tilt = 7
        else:
            self.y_tilt = 4

    def tilt(self):
        return (self.x_tilt, self.y_tilt)


class Gaze():
    def __init__(self, w, h):
        #   Nine directions:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        self.direction = 4
        self.boxs = {
            #   [(x1, y1), (x2, y2)]
            0 : [(0, 0), (w//2, h//2)],
            1 : [(w//4, 0), (int(w*0.75), h//2)],
            2 : [(w//2, 0), (w, h//2)],
            3 : [(0, h//4), (w//2, int(h*0.75))],
            4 : [(w//4, h//4), (int(w*0.75), int(h*0.75))],
            5 : [(w//2, h//4), (w, int(h*0.75))],
            6 : [(0, h//2), (w//2, h)],
            7 : [(w//4, h//2), (int(w*0.75), h)],
            8 : [(w//2, h//2), (w, h)],
        }

    def draw_gaze(self, frame):
        box = self.boxs[self.direction]
        overlay = frame.copy()
        cv2.rectangle(overlay, box[0], box[1], (255,0,0), -1)
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, 0)


def check_corners(eye):
    #
    #   0 - 2
    #   - - -
    #   6 - 8
    #
    if eye.pupil[0] < eye.center_xy[0]:
        if eye.pupil[1] < eye.center_xy[1]:
            return 0
        if eye.pupil[1] > eye.center_xy[1]:
            return 6
    if eye.pupil[0] > eye.center_xy[0]:
        if eye.pupil[1] > eye.center_xy[1]:
            return 8
        if eye.pupil[1] < eye.center_xy[1]:
            return 2
    return 4


def determine_eye_gaze(eyes):
    # How far away is the pupil from the center of the detected eye point
    #
    #          -hr                    1
    #  -wr  center_xy  +wr          3 4 5
    #          +hr                    7
    #
    for eye in eyes:
        if eye.closed:
            return 7
        a = eye.box[2] * 0.25
        b = eye.box[3] * 0.25
        radius = math.sqrt( a**2 + b**2 )
        pupil = dist.euclidean(eye.pupil, eye.center_xy)
        if pupil > radius:
            return check_corners(eye)
        wr = int(eye.box[2] * 0.20)
        hr = int(eye.box[3] * 0.30)
        if eye.pupil[0] < eye.center_xy[0] - wr:
            return 3
        if eye.pupil[0] > eye.center_xy[0] + wr:
            return 5
        if eye.pupil[1] < eye.center_xy[1] - hr:
            return 1
        if eye.pupil[1] > eye.center_xy[1] + hr:
            return 7
    return 4


def gaze_direction(eye_gaze, FT, G):
    d = 4
    # Facing forward
    if FT.tilt() == (4, 4):
        d = eye_gaze
    # Facing down
    elif FT.tilt() == (4, 7):
        if eye_gaze == 1:
            d = 4
        else:
            d = 7
    # Facing left
    elif FT.tilt() == (5, 4):
        if eye_gaze == 3:
            d = 4
        elif eye_gaze == 1:
            d = 2
        else:
            d = 5
    # Facing right
    elif FT.tilt() == (3, 4):
        if eye_gaze == 5:
            d = 4
        elif eye_gaze == 1:
            d = 0
        else:
            d = 3
    # Facing right and down
    elif FT.tilt() == (3, 7):
        if eye_gaze == 5:
            d = 7
        else:
            d = 6
    # Facing left and down
    elif FT.tilt() == (5, 7):
        if eye_gaze == 3:
            d = 7
        else:
            d = 8
    G.direction = d


def draw_eyes(frame, eyes):
    trackeye.find_colors(eyes)
    for eye in eyes:
        #cv2.circle(frame, (eye.center_xy), 3, (255, 255, 255), -1)
        #cv2.rectangle(
            #frame,
            #(eye.box[0], eye.box[1]),
            #(eye.box[0]+eye.box[2], eye.box[1]+eye.box[3]),
            #(255,255,255), 1)
        if not eye.closed:
            #cv2.rectangle(
                #frame,
                #(eye.box[0], eye.box[1]),
                #eye.center_xy,
                #(255,255,255), 1)
            #cv2.rectangle(
                #frame,
                #eye.center_xy,
                #(eye.box[0]+eye.box[2], eye.box[1]+eye.box[3]),
                #(255,255,255), 1)
            cv2.circle(frame, (eye.pupil), 2, (0, 0, 255), -1)
            if eye.LR == "L":
                cv2.rectangle(frame, (w-128, 0), (w-64, 64), eye.BGR, -1)
            else:
                cv2.rectangle(frame, (w-64, 0), (w, 64), eye.BGR, -1)
    if not eyes[0].closed or not eyes[1].closed:
        cv2.putText(frame, "(eye colors)", (w-96, 56), 0, 0.42, (255,255,255), 1)
    return frame


def read_vid():
    FT = Face_Tilt()
    G = Gaze(w, h)
    while True:
        check, frame = vs.read()
        if not check or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for i, face in enumerate(faces):
            shape = trackeye.shape_to_np(predictor(gray, face))
            FT.determine_x_tilt(shape)
            FT.determine_y_tilt(shape)
            eyes = trackeye.extract_eyes(predictor, detector, gray)
            trackeye.find_centers(eyes, frame)
            eye_gaze = determine_eye_gaze(eyes)
            gaze_direction(eye_gaze, FT, G)
            frame = G.draw_gaze(frame)
            frame = draw_eyes(frame, eyes)
            #for x, y in shape:
                #cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            writer.write(frame)
            break
        cv2.imshow("", frame)
        #cv2.imshow("", np.hstack([eyes[0].img, eyes[0].thresh]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def vid_info(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps -= 5
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, fps


if __name__ == "__main__":
    in_vid = "vids/m0.mp4"
    w, h, fps = vid_info(in_vid)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(f"tmp0.mp4", fourcc, fps, (w,h), True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    vs = cv2.VideoCapture(in_vid)
    read_vid()
    vs.release()



##
