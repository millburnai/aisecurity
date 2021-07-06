"""Graphics util.
"""

import threading

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] '{e}'. Ignore if Realsense is not set up")



NVARGUS = "nvarguscamerasrc ! video/x-raw(memory:NVMM), " \
          "width=(int)1280, height=(int)720, format=(string)NV12, " \
          "framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, " \
          "width=(int){}, height=(int){}, format=(string)BGRx ! " \
          "videoconvert ! video/x-raw, format=(string)BGR ! appsink"


class Camera:
    # https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/camera.py

    def __init__(self, width=640, height=360, fps=30, dev=0, threaded=False):
        self.width = width
        self.height = height
        self.dev = dev
        self.threaded = threaded
        self.fps = fps

        self.thread_running = False
        self.retval = False
        self.img_handle = None
        self.thread = None
        self.cap = None

        self._open()
        self._start()

    def _open(self):
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, 
                                 rs.format.bgr8, self.fps)
            self.pipeline = rs.pipeline()
            self.pipeline.start(config)
            assert self.pipeline.get_active_profile(), "video capture failed to initialize"

        except:
            try:
                gstreamer_pipeline = NVARGUS.format(self.width, self.height)
                self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
                assert self.cap.isOpened(), "video capture failed to initialize"

            except AssertionError:
                self.cap = cv2.VideoCapture(self.dev)
                assert self.cap.isOpened(), "video capture failed to initialize"

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_realsense(self):
        ret = True
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            ret = False
            return ret, None
        else:
            color_image = np.array(color_frame.get_data())
            return ret, color_image

    def _start(self):
        def grab_img(cam):
            while cam.thread_running:
                try:
                    cam.retval, cam.img_handle = self.read_realsense()
                except:
                    cam.retval, cam.img_handle = cam.cap.read()
            cam.thread_running = False

        if self.threaded:
            assert not self.thread_running, "thread is already running"
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def read(self):
        if self.threaded:
            return self.retval, self.img_handle
        else:
            try:
                return self.read_realsense()
            except:
                return self.cap.read()

    def release(self):
        if self.threaded:
            self.thread_running = False
            self.thread.join()
        try:
            self.cap.release()
        except:
            self.pipeline.stop()


class GraphicsRenderer:

    def __init__(self, width=640, height=360, resize=1., margin=10,
                 font=cv2.FONT_HERSHEY_DUPLEX):
        self.width = width
        self.height = height
        self.resize = resize
        self.margin = margin
        self.font = font

        # works for 6.25e4 pixel video cature to 1e6 pixel video capture
        self.line_thickness = round(1e-6 * width * height + 1.5)
        self.radius = round((1e-6 * width * height + 1.5) / 2.)
        self.font_size = 4.5e-7 * width * height + 0.5

    def add_box_and_label(self, frame, origin, corner,
                          best_match, color, thickness=1):
        # bounding box
        cv2.rectangle(frame, origin, corner, color, self.line_thickness)

        # label box
        label = best_match.replace("_", " ").title()

        (width, height), __ = cv2.getTextSize(label, self.font,
                                              self.font_size, thickness)

        box_x = max(corner[0], origin[0] + width + 6)
        cv2.rectangle(frame, (origin[0], corner[1] - 35),
                      (box_x, corner[1]), color, cv2.FILLED)

        # label
        cv2.putText(frame, label, (origin[0] + 6, corner[1] - 6),
                    self.font, self.font_size, (255, 255, 255), thickness)

    def add_features(self, overlay, features, color):
        cv2.circle(overlay, (features["left_eye"]),
                   self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["right_eye"]),
                   self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["nose"]),
                   self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["mouth_left"]),
                   self.radius, color, self.line_thickness)
        cv2.circle(overlay, (features["mouth_right"]),
                   self.radius, color, self.line_thickness)

        cv2.line(overlay, features["left_eye"], features["nose"],
                 color, self.radius)
        cv2.line(overlay, features["right_eye"], features["nose"],
                 color, self.radius)
        cv2.line(overlay, features["mouth_left"], features["nose"],
                 color, self.radius)
        cv2.line(overlay, features["mouth_right"], features["nose"],
                 color, self.radius)

    def add_fps(self, frame, elapsed, thickness=2):
        text = "FPS: {}".format(round(1000. / elapsed, 2))

        x, y = 10, 20
        rgb = [255 * round((255 - np.mean(frame[:x, :y])) / 255)] * 3

        cv2.putText(frame, text, (x, y), self.font,
                    self.font_size, rgb, thickness)

    def add_graphics(self, frame, person, is_recognized, best_match, elapsed):
        def get_color():
            if not is_recognized:
                return 0, 0, 255  # red
            elif "visitor" in best_match:
                return 218, 112, 214  # purple (actually more of an "orchid")
            else:
                return 0, 255, 0  # green

        if person is not None:
            features = person["keypoints"]
            x, y, height, width = person["box"]

            if self.resize != 1.:
                scale_factor = 1. / self.resize

                if features:
                    scale = lambda x: tuple(round(element * scale_factor)
                                            for element in x)
                    features = {feature: scale(features[feature])
                                for feature in features}

                scale = lambda *xs: tuple(int(round(x * scale_factor))
                                          for x in xs)
                x, y, height, width = scale(x, y, height, width)

            color = get_color()
            origin = (x - self.margin // 2, y - self.margin // 2)
            corner = (x + height + self.margin // 2,
                      y + width + self.margin // 2)

            if features:
                overlay = frame.copy()
                self.add_features(overlay, features, color)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            text = best_match if is_recognized else ""
            self.add_box_and_label(frame, origin, corner,
                                   text, color, thickness=1)

        self.add_fps(frame, elapsed)
