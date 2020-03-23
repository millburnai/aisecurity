"""

"aisecurity.face.detection"

Haarcascade or MTCNN face detection.

"""

import cv2
from mtcnn import MTCNN as MTCNNBackend

from aisecurity.utils.paths import CONFIG_HOME


# GLOBALS ALLOC
MTCNN = None
HAARCASCADE = None

PARAMS = {}


# FUNCS
def detector_init(min_face_size=20, filepath=CONFIG_HOME+"/models/haarcascae_frontalface_default.xml", **kwargs):
    global MTCNN, HAARCASCADE, PARAMS

    MTCNN = MTCNNBackend(min_face_size=min_face_size, **kwargs)
    HAARCASCADE = cv2.CascadeClassifier(filepath)

    PARAMS["min_face_size"] = min_face_size
    PARAMS.update(kwargs)


def detect_faces(img, mode="mtcnn"):
    assert MTCNN and HAARCASCADE, "call detector_init() before using detect_faces()"

    result = []

    if mode == "mtcnn":
        result = MTCNN.detect_faces(img)

    if mode == "haarcascade" or (mode == "both" and not result):
        min_face_size = PARAMS["min_face_size"]
        faces = HAARCASCADE.detectMultiScale(img, scaleFactor=1.1, minSize=(min_face_size, min_face_size))

        for (x, y, width, height) in faces:
            result.append({
                "box": [x, y, width, height],
                "keypoints": None,
                "confidence": 1.0
            })

    else:
        raise ValueError("'{}' not a supported mode (supported are 'mtcnn' and 'haarcascade')".format(mode))

    return result
