"""Graphics utils.

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
        self.retval = False
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
                cam.retval, cam.img_handle = cam.cap.read()

            cam.thread_running = False

        assert not self.thread_running, "thread is already running"
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def read(self):
        return self.retval, self.img_handle

    def release(self):
        self.thread_running = False
        self.thread.join()
        self.cap.release()


################################ Graphics ###############################
class GraphicsController:

    def __init__(self, width=640, height=360, resize=1., margin=10, font=cv2.FONT_HERSHEY_DUPLEX):
        self.width = width
        self.height = height
        self.resize = resize
        self.margin = margin
        self.font = font

        # works for 6.25e4 pixel video cature to 1e6 pixel video capture
        self.line_thickness = round(1e-6 * width * height + 1.5)
        self.radius = round((1e-6 * width * height + 1.5) / 2.)
        self.font_size = 4.5e-7 * width * height + 0.5


    # HELPERS
    @staticmethod
    def _get_color(is_recognized, best_match):
        if not is_recognized:
            return 0, 0, 255  # red
        elif "visitor" in best_match:
            return 218, 112, 214  # purple (actually more of an "orchid")
        else:
            return 0, 255, 0  # green

    def _add_box_and_label(self, frame, origin, corner, best_match, color, thickness=1):
        # bounding box
        cv2.rectangle(frame, origin, corner, color, self.line_thickness)

        # label box
        label = best_match.replace("_", " ").title()

        (width, height), __ = cv2.getTextSize(label, self.font, self.font_size, thickness)

        box_x = max(corner[0], origin[0] + width + 6)
        cv2.rectangle(frame, (origin[0], corner[1] - 35), (box_x, corner[1]), color, cv2.FILLED)

        # label
        cv2.putText(frame, label, (origin[0] + 6, corner[1] - 6), self.font, self.font_size, (255, 255, 255), thickness)

    def _add_features(self, overlay, features, color):
        cv2.circle(overlay, (features["left_eye"]), self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["right_eye"]), self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["nose"]), self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["mouth_left"]), self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["mouth_right"]), self.radius, color, self.line_thickness)

        cv2.line(overlay, features["left_eye"], features["nose"], color, self.radius)
        cv2.line(overlay, features["right_eye"], features["nose"], color, self.radius)
        cv2.line(overlay, features["mouth_left"], features["nose"], color, self.radius)
        cv2.line(overlay, features["mouth_right"], features["nose"], color, self.radius)

    def _add_fps(self, frame, elapsed, thickness=2):
        text = "FPS: {}".format(round(1000. / elapsed, 2))  # elapsed is in ms, so *1000.

        x, y = 10, 20
        rgb = [255 * round((255 - np.mean(frame[:x, :y])) / 255)] * 3

        cv2.putText(frame, text, (x, y), self.font, self.font_size, rgb, thickness)


    # ADD GRAPHICS
    def add_graphics(self, frame, person, is_recognized, best_match, elapsed):
        if person is not None:
            features = person["keypoints"]
            x, y, height, width = person["box"]

            if self.resize != 1.:
                scale_factor = 1. / self.resize

                if features:
                    scale = lambda x: tuple(round(element * scale_factor) for element in x)
                    features = {feature: scale(features[feature]) for feature in features}

                scale = lambda *xs: tuple(int(round(x * scale_factor)) for x in xs)
                x, y, height, width = scale(x, y, height, width)

            color = self._get_color(is_recognized, best_match)
            origin = (x - self.margin // 2, y - self.margin // 2)
            corner = (x + height + self.margin // 2, y + width + self.margin // 2)

            if features:
                overlay = frame.copy()
                self._add_features(overlay, features, color)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            text = best_match if is_recognized else ""
            self._add_box_and_label(frame, origin, corner, text, color, thickness=1)

        self._add_fps(frame, elapsed)
