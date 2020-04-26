"""

"aisecurity.visuals.graphics"

Graphics utils.

"""

import threading

import cv2
import numpy as np


################################ Camera ###############################
class Camera:
    # https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/camera.py

    def __init__(self, width=640, height=360, dev=0):
        self.width = width
        self.height = height
        self.dev = dev

        self.thread_running = False
        self.img_handle = None
        self.thread = None

        self._open()
        self._start()

    def _open(self):
        try:
            gstreamer_pipeline = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, "
                                  "format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! "
                                  "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! "
                                  "video/x-raw, format=(string)BGR ! appsink").format(self.width, self.height)
            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            assert self.cap.isOpened(), "video capture failed to initialize"

        except AssertionError:
            self.cap = cv2.VideoCapture(self.dev)
            assert self.cap.isOpened(), "video capture failed to initialize"

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _start(self):
        def grab_img(cam):
            while cam.thread_running:
                _, cam.img_handle = cam.cap.read()

            cam.thread_running = False

        assert not self.thread_running, "thread is already running"
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def read(self):
        return self.img_handle

    def release(self):
        self.thread_running = False
        self.thread.join()
        self.cap.release()


################################ Graphics ###############################
def add_graphics(frame, person, width, height, is_recognized, best_match, resize, elapsed, margin=10):
    """Adds graphics to a frame

    :param frame: frame as array
    :param person: MTCNN detection dict
    :param width: width of frame
    :param height: height of frame
    :param is_recognized: whether face was recognized or not
    :param best_match: best match from database
    :param resize: resize scale factor, from 0. to 1.
    :param elapsed: time it took to run face detection and recognition
    :param margin: crop margin for face detection (default: 10)

    """

    line_thickness = round(1e-6 * width * height + 1.5)
    radius = round((1e-6 * width * height + 1.5) / 2.)
    font_size = 4.5e-7 * width * height + 0.5
    # works for 6.25e4 pixel video cature to 1e6 pixel video capture

    def get_color(is_recognized, best_match):
        if not is_recognized:
            return 0, 0, 255  # red
        elif "visitor" in best_match:
            return 218, 112, 214  # purple (actually more of an "orchid")
        else:
            return 0, 255, 0  # green

    def add_box_and_label(frame, origin, corner, color, line_thickness, best_match, font_size, thickness):
        # bounding box
        cv2.rectangle(frame, origin, corner, color, line_thickness)

        # label box
        label = best_match.replace("_", " ").title()
        font = cv2.FONT_HERSHEY_DUPLEX

        (width, height), __ = cv2.getTextSize(label, font, font_size, thickness)

        box_x = max(corner[0], origin[0] + width + 6)
        cv2.rectangle(frame, (origin[0], corner[1] - 35), (box_x, corner[1]), color, cv2.FILLED)

        # label
        cv2.putText(frame, label, (origin[0] + 6, corner[1] - 6), font, font_size, (255, 255, 255), thickness)


    def add_features(overlay, features, radius, color, line_thickness):
        cv2.circle(overlay, (features["left_eye"]), radius, color, line_thickness)
        cv2.circle(overlay, (features["right_eye"]), radius, color, line_thickness)
        cv2.circle(overlay, (features["nose"]), radius, color, line_thickness)
        cv2.circle(overlay, (features["mouth_left"]), radius, color, line_thickness)
        cv2.circle(overlay, (features["mouth_right"]), radius, color, line_thickness)

        cv2.line(overlay, features["left_eye"], features["nose"], color, radius)
        cv2.line(overlay, features["right_eye"], features["nose"], color, radius)
        cv2.line(overlay, features["mouth_left"], features["nose"], color, radius)
        cv2.line(overlay, features["mouth_right"], features["nose"], color, radius)

    def add_fps(frame, elapsed, font_size, thickness):
        text = "FPS: {}".format(round(1000. / elapsed, 2))  # elapsed is in ms, so *1000.

        x, y = 10, 20
        font = cv2.FONT_HERSHEY_DUPLEX
        rgb = [255. - np.mean(frame[:x, :y])] * 3

        cv2.putText(frame, text, (x, y), font, font_size, rgb, thickness)

    if person is not None:
        features = person["keypoints"]
        x, y, height, width = person["box"]

        if resize:
            scale_factor = 1. / resize

            if features:
                scale = lambda x: tuple(round(element * scale_factor) for element in x)
                features = {feature: scale(features[feature]) for feature in features}

            scale = lambda *xs: tuple(int(round(x * scale_factor)) for x in xs)
            x, y, height, width = scale(x, y, height, width)

        color = get_color(is_recognized, best_match)
        origin = (x - margin // 2, y - margin // 2)
        corner = (x + height + margin // 2, y + width + margin // 2)

        if features:
            overlay = frame.copy()
            add_features(overlay, features, radius, color, line_thickness)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        text = best_match if is_recognized else ""
        add_box_and_label(frame, origin, corner, color, line_thickness, text, font_size, thickness=1)

    add_fps(frame, elapsed, font_size, thickness=2)
