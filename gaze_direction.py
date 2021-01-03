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


class Face_Tilt:
    def __init__(self):
        self.x_tilt = "" # left right forward or L R F
        self.y_tilt = "" # up down forward or U D F
        self.upr_lip = 0
        self.chin = 0

    def determine_x_tilt(self, shape):
        right = dist.euclidean(shape[30], shape[31])
        left = dist.euclidean(shape[30], shape[35])
        if right > left + left * 0.80:
            self.x_tilt = "L"
        elif left > right + right * 0.80:
            self.x_tilt = "R"
        else:
            self.x_tilt = "F"

    def determine_y_tilt(self, shape):
        upr_lip = dist.euclidean(shape[33], shape[51])
        chin = dist.euclidean(shape[8], shape[57])
        if upr_lip > self.upr_lip:
            self.upr_lip = upr_lip
        if chin > self.chin:
            self.chin = chin
        if upr_lip < self.upr_lip * 0.70 and chin < self.chin * 0.70:
            self.y_tilt = "D"
        else:
            self.y_tilt = "F"


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




def eyegaze(eyes):
    #
    #          -hr
    #  -wr  center_xy  +wr
    #          +hr
    #
    left = eyes[0]
    right = eyes[1]
    left_w = left.box[2]
    right_w = right.box[2]
    left_wr = int(left_w * 0.20)
    right_wr = int(right_w * 0.20)
    left_h = left.box[3]
    right_h = left.box[3]
    left_hr = int(left_h * 0.30)
    right_hr = int(right_h * .30)
    if left.closed or right.closed:
        return "D"
    if left.pupil[0] < left.center_xy[0] - left_wr:
        return "R"
    if right.pupil[0] < right.center_xy[0] - right_wr:
        return "R"
    if left.pupil[0] > left.center_xy[0] + left_wr:
        return "L"
    if right.pupil[0] > right.center_xy[0] + right_wr:
        return "L"
    if left.pupil[1] < left.center_xy[1] - left_hr:
        return "U"
    if right.pupil[1] < right.center_xy[1] - right_hr:
        return "U"
    if left.pupil[1] > left.center_xy[1] + left_hr:
        return "D"
    if right.pupil[1] > right.center_xy[1] + right_hr:
        return "D"
    else:
        return "F"


def gaze_direction(eye_gaze, FT, G):
    # Is there a cooler way to do this with states?
    d = 4
    if FT.x_tilt == "F" and FT.y_tilt == "F":
        if eye_gaze == "U":
            d = 1
        elif eye_gaze == "R":
            d = 3
        elif eye_gaze == "F":
            d = 4
        elif eye_gaze == "L":
            d = 5
        else:
            d = 7
    elif FT.x_tilt == "R" and FT.y_tilt == "D":
        if eye_gaze == "L":
            d = 7
        else:
            d = 6
    elif FT.x_tilt == "F" and FT.y_tilt == "D":
        if eye_gaze == "U":
            d = 4
        else:
            d = 7
    elif FT.x_tilt == "L" and FT.y_tilt == "D":
        if eye_gaze == "R":
            d = 7
        else:
            d = 8
    elif FT.x_tilt == "R" and FT.y_tilt == "F":
        if eye_gaze == "L":
            d = 4
        elif eye_gaze == "U":
            d = 0
        else:
            d = 3
    elif FT.x_tilt == "L" and FT.y_tilt == "F":
        if eye_gaze == "R":
            d = 4
        elif eye_gaze == "U":
            d = 2
        else:
            d = 5
    G.direction = d


def draw_eyes(frame, eyes):
    for eye in eyes:
        #cv2.circle(frame, (eye.center_xy), 4, (255, 255, 255), -1)
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
            eye_gaze = eyegaze(eyes)
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


def meta_info(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps -= 2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, fps


if __name__ == "__main__":
    in_vid = "vids/m0.mp4"
    w, h, fps = meta_info(in_vid)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(f"tmp0.mp4", fourcc, fps, (w,h), True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    vs = cv2.VideoCapture(in_vid)
    read_vid()
    vs.release()



##
