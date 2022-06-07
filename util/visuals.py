"""Graphics util.
"""

from abc import ABC, abstractmethod
import threading

import cv2
import numpy as np

from util.common import HAS_RS

try:
    import pyrealsense2 as rs
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] '{e}'. Ignore if Realsense is not set up")
    HAS_RS = False


def Camera(**kwargs):
    if HAS_RS:
        return RSCapture(**kwargs)
    return WebcamCapture(**kwargs)


class CVThread(ABC):
    def stop(self):
        self.stopped = True

    def _setup(self, uid, tname, args):
        self.uid = uid
        self.tname = tname
        self.stopped = False
        self._next()

        thread = threading.Thread(target=self._update, name=tname, args=args)
        thread.daemon = True
        thread.start()

    def _update(self):
        while True:
            if self.stopped:
                self._cleanup()
                return
            self._next()

    @abstractmethod
    def _next(self):
        """Sets nexts frames"""

    @abstractmethod
    def _cleanup(self):
        """Cleans up on termination"""

    @abstractmethod
    def read(self):
        """Returns next frames"""

    def release(self):
        self.stopped = True


class WebcamCapture(CVThread):
    def __init__(self, src=0, *args):
        self.cap = cv2.VideoCapture(src)
        self._setup(src, f"VS-{src}", args)

    def _next(self):
        self.retval, self.frame = self.cap.read()

    def _cleanup(self):
        pass

    def read(self):
        return self.retval, self.frame


class RSCapture(CVThread):
    def __init__(self, uid=0, *args):
        self.stream = rs.pipeline()

        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color)

        self.stream.start(rs_cfg)
        self._setup(uid, f"RS-{uid}", args)

    def _next(self):
        frames = self.stream.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.frame = frame

    def _cleanup(self):
        self.stream.stop()

    def read(self):
        return True, self.frame


class GraphicsRenderer:
    def __init__(
        self, width=640, height=360, resize=1.0, margin=10, font=cv2.FONT_HERSHEY_DUPLEX
    ):
        self.width = width
        self.height = height
        self.resize = resize
        self.margin = margin
        self.font = font

        # works for 6.25e4 pixel video cature to 1e6 pixel video capture
        self.line_thickness = round(1e-6 * width * height + 1.5)
        self.radius = round((1e-6 * width * height + 1.5) / 2.0)
        self.font_size = 4.5e-7 * width * height + 0.5

    def add_box_and_label(self, frame, origin, corner, best_match, color, thickness=1):
        # bounding box
        cv2.rectangle(frame, origin, corner, color, self.line_thickness)

        # label box
        label = best_match.replace("_", " ").title()

        (width, height), __ = cv2.getTextSize(
            label, self.font, self.font_size, thickness
        )

        box_x = max(corner[0], origin[0] + width + 6)
        cv2.rectangle(
            frame, (origin[0], corner[1] - 35), (box_x, corner[1]), color, cv2.FILLED
        )

        # label
        cv2.putText(
            frame,
            label,
            (origin[0] + 6, corner[1] - 6),
            self.font,
            self.font_size,
            (255, 255, 255),
            thickness,
        )

    def add_features(self, overlay, features, color):
        cv2.circle(
            overlay, (features["left_eye"]), self.radius, color, self.line_thickness
        )
        cv2.circle(
            overlay, (features["right_eye"]), self.radius, color, self.line_thickness
        )
        cv2.circle(overlay, (features["nose"]), self.radius, color, self.line_thickness)
        cv2.circle(
            overlay, (features["mouth_left"]), self.radius, color, self.line_thickness
        )
        cv2.circle(
            overlay, (features["mouth_right"]), self.radius, color, self.line_thickness
        )

        cv2.line(overlay, features["left_eye"], features["nose"], color, self.radius)
        cv2.line(overlay, features["right_eye"], features["nose"], color, self.radius)
        cv2.line(overlay, features["mouth_left"], features["nose"], color, self.radius)
        cv2.line(overlay, features["mouth_right"], features["nose"], color, self.radius)

    def add_fps(self, frame, elapsed, thickness=2):
        text = "FPS: {}".format(round(1000.0 / elapsed, 2))

        x, y = 10, 20
        rgb = [255 * round((255 - np.mean(frame[:x, :y])) / 255)] * 3

        cv2.putText(frame, text, (x, y), self.font, self.font_size, rgb, thickness)

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

            if self.resize != 1.0:
                scale_factor = 1.0 / self.resize

                if features:
                    scale = lambda x: tuple(
                        round(element * scale_factor) for element in x
                    )
                    features = {
                        feature: scale(features[feature]) for feature in features
                    }

                scale = lambda *xs: tuple(int(round(x * scale_factor)) for x in xs)
                x, y, height, width = scale(x, y, height, width)

            color = get_color()
            origin = (x - self.margin // 2, y - self.margin // 2)
            corner = (x + height + self.margin // 2, y + width + self.margin // 2)

            if features:
                overlay = frame.copy()
                self.add_features(overlay, features, color)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            text = best_match if is_recognized else ""
            self.add_box_and_label(frame, origin, corner, text, color, thickness=1)

        #self.add_fps(frame, elapsed)
