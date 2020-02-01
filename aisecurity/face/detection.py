"""

"aisecurity.face.detection"

Haarcascade or MTCNN face detection.

"""

from aisecurity.utils.events import timer
from aisecurity.utils.paths import CONFIG_HOME

import cv2
from mtcnn.mtcnn import MTCNN


# GLOBALS ALLOC
FACE_DETECTORS = {
    "mtcnn": None,
    "haarcascade": None
}


# FACE DETECTOR
class FaceDetector:

    MODES = ("mtcnn", "haarcascade")

    def __init__(self, mode, filepath=CONFIG_HOME+"/models/haarcascade_frontalface_default.xml", min_face_size=20):
        assert mode in self.MODES, "supported modes are {}".format(self.MODES)

        self.mode = mode
        self.min_face_size = min_face_size

        if self.mode == "mtcnn":
            self.detector = MTCNN(min_face_size=self.min_face_size)
        elif self.mode == "haarcascade":
            self.detector = cv2.CascadeClassifier(filepath)

    @timer(message="Detection time")
    def detect_faces(self, img):
        if self.mode == "mtcnn":
            result = self.detector.detect_faces(img)

        elif self.mode == "haarcascade":
            result = []

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.equalizeHist(img)

            faces = self.detector.detectMultiScale(
                img, scaleFactor=1.2, minSize=(self.min_face_size, self.min_face_size)
            )

            for (x, y, width, height) in faces:
                result.append({
                    "box": [x, y, width, height],
                    "keypoints": None,
                    "confidence": 1.0
                })

        return result


# FUNCS
def detector_init(**kwargs):
    global FACE_DETECTORS

    FACE_DETECTORS["mtcnn"] = FaceDetector("mtcnn", **kwargs)
    FACE_DETECTORS["haarcascade"] = FaceDetector("haarcascade", **kwargs)


def detect_faces(img, mode="mtcnn"):
    return FACE_DETECTORS[mode].detect_faces(img)
