# gaze_direction.py
# $ python3 gaze_direction.py
# Jan 1 2021
#
# With an input a video file...
# Get the gaze direction, assuming only one person in the frame.
# Calculate the face tilt direction and the eye gaze direction,
# and use these to determine relatively which direction the person is looking


import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import trackeye


class Face_Tilt:
    def __init__(self):
        self.x_tilt = "" # left right forward or L R F
        self.y_tilt = "" # up down forward or U D F
        self.upr_lip = 0
        self.chin = 0

    def get_tilt(self):
        return self.x_tilt, self.y_tilt

    def __repr__(self):
        return f"{self.get_tilt()}"


class Gaze():
    def __init__(self, w, h):
        self.prev = 0
        self.direction = 0
        #   Nine directions:
        #   0 1 2
        #   3 4 5
        #   6 7 8
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


def x_tilt(shape, FT):
    right = dist.euclidean(shape[30], shape[31])
    left = dist.euclidean(shape[30], shape[35])
    if right > left + left * 0.5:
        FT.x_tilt = "L"
    elif left > right + right * 0.5:
        FT.x_tilt = "R"
    else:
        FT.x_tilt = "F"


def y_tilt(shape, FT):
    upr_lip = dist.euclidean(shape[33], shape[51])
    chin = dist.euclidean(shape[8], shape[57])
    if upr_lip > FT.upr_lip:
        FT.upr_lip = upr_lip
    if chin > FT.chin:
        FT.chin = chin
    if upr_lip < FT.upr_lip * 0.8 and chin < FT.chin * 0.8:
        FT.y_tilt = "D"
    else:
        FT.y_tilt = "F"


def eyegaze(eyes):
    left = eyes[0]
    right = eyes[1]
    if left.closed or right.closed:
        return "D"
    #if left.pupil[0] < left.center_xy[0] - (left.center_xy[0]//4):
        #if right.pupil[0] < right.center_xy[0] - (right.center_xy[0]//4):
            #return "R"
    if left.pupil[0] < left.center_xy[0]:
        if right.pupil[0] < right.center_xy[0]:
            return "R"
    if left.pupil[0] > left.center_xy[0]:
        if right.pupil[0] > right.center_xy[0]:
            return "L"
    if left.pupil[1] >= left.center_xy[1] + (max(right.box[2]//4,1)):
        if right.pupil[1] >= right.center_xy[1] + (max(right.box[2]//4,1)):
            return "D"
    else:
        return "F"


def gaze_direction(eye_gaze, FT, G):
    # Is there a cooler way to do this with states?
    d = 4
    if FT.x_tilt == "F" and FT.y_tilt == "F":
        if eye_gaze == "R":
            d = 3
        elif eye_gaze == "F":
            d = 4
        elif eye_gaze == "L":
            d = 5
    elif FT.x_tilt == "R" and FT.y_tilt == "D":
        if eye_gaze == "L":
            d = 7
        else:
            d = 6
    elif FT.x_tilt == "F" and FT.y_tilt == "D":
        d = 7
    elif FT.x_tilt == "L" and FT.y_tilt == "D":
        if eye_gaze == "R":
            d = 7
        else:
            d = 8
    elif FT.x_tilt == "R" and FT.y_tilt == "F":
        if eye_gaze == "L":
            d = 4
        else:
            d = 3
    elif FT.x_tilt == "L" and FT.y_tilt == "F":
        if eye_gaze == "R":
            d = 4
        else:
            d = 5
    G.direction = d


def draw(frame, G):
    #if G.direction != G.prev:
        #G.prev = G.direction
        #return frame
    box = G.boxs[G.direction]
    overlay = frame.copy()
    cv2.rectangle(overlay, box[0], box[1], (255,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, 0)
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
            x_tilt(shape, FT)
            y_tilt(shape, FT)
            eyes = trackeye.extract_eyes(predictor, detector, gray)
            trackeye.find_centers(eyes, frame)
            eye_gaze = eyegaze(eyes)
            gaze_direction(eye_gaze, FT, G)
            frame = draw(frame, G)
            for eye in eyes:
                cv2.circle(frame, (eye.pupil), 2, (0, 0, 255), -1)
            #cv2.putText(frame, f"{eye_gaze}-{FT}", (10,25), 0, 1, (0,0,255), 1)
            #for x, y in shape:
                #cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            writer.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def meta_info(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, fps


if __name__ == "__main__":
    in_vid = "vids/jimmy5.mp4"
    w, h, fps = meta_info(in_vid)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(f"tmp0.mp4", fourcc, fps, (w,h), True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    vs = cv2.VideoCapture(in_vid)
    read_vid()
    vs.release()



##
