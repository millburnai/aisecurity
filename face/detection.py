"""Haarcascade or MTCNN face detection.
"""

import os
import sys
from timeit import default_timer as timer

import cv2
from mtcnn import MTCNN
import numpy as np

sys.path.insert(1, "../")
from utils.paths import config_home


def load_trt_mtcnn_plugin(module, engine_path="mtcnn/det{}.engine"):
    engine_paths = [os.path.join(module, engine_path).format(net + 1)
                    for net in range(3)]

    assert all(os.path.exists(net) for net in engine_paths), \
        "trt-mtcnn engines not found"
    sys.path.insert(0, module)

    from trt_mtcnn_main import TrtMTCNNWrapper
    return TrtMTCNNWrapper(*engine_paths)


class FaceDetector:
    MODES = ["mtcnn", "haarcascade", "trt-mtcnn"]

    def __init__(self, mode, img_shape=(160, 160), alpha=0.9,
                 plugin_path=config_home + "/trt_mtcnn_plugin", **kwargs):
        assert any(det in mode for det in self.MODES), \
            "supported modes are 'mtcnn', 'haarcascade', and 'trt-mtcnn'"

        self.mode = mode
        self.alpha = alpha
        self.img_shape = tuple(img_shape)
        self.kwargs = kwargs

        if "min_face_size" not in self.kwargs:
            self.kwargs["min_face_size"] = 20

        self.mtcnn = MTCNN(min_face_size=self.kwargs["min_face_size"])
        # TODO: figure out why we need this for the threaded cam to work

        if "trt-mtcnn" in mode:
            self.trt_mtcnn = load_trt_mtcnn_plugin(plugin_path)

        if "haarcascade" in mode:
            self.haarcascade = cv2.CascadeClassifier(plugin_path)

    def detect_faces(self, img):
        result = []

        if "trt-mtcnn" in self.mode:
            minsize = max(40, int(self.kwargs["min_face_size"]))
            result = self.trt_mtcnn.detect_faces(img, minsize=minsize)

        if "mtcnn" in self.mode and "trt-mtcnn" not in self.mode:
            result = self.mtcnn.detect_faces(img)

        no_result = (not result or result[0]["confidence"] < self.alpha)
        if "haarcascade" in self.mode and no_result:
            min_face_size = int(self.kwargs["min_face_size"])
            faces = self.haarcascade.detectMultiScale(
                img, scaleFactor=1.1, minSize=(min_face_size, min_face_size)
            )
            for x, y, width, height in faces:
                result.append({
                    "box": [x, y, width, height],
                    "keypoints": None,
                    "confidence": 1.
                })

        return result

    def crop_face(self, img_bgr, margin, rotations=None):
        def crop_and_rotate(img, img_shape, face_coords, rotation_angle):
            x, y, width, height = face_coords
            img = img[y - margin // 2:y + height + margin // 2,
                      x - margin // 2:x + width + margin // 2, :]

            resized = cv2.resize(img, img_shape)
            if rotation_angle == 0:
                return resized
            elif rotation_angle == -1:
                return cv2.flip(resized, 1)
            else:
                rot = tuple(np.array(resized.shape[1::-1]) / 2)
                mat = cv2.getRotationMatrix2D(rot, rotation_angle, 1.)
                return cv2.warpAffine(resized, mat, resized.shape[1::-1],
                                      flags=cv2.INTER_LINEAR)

        start = timer()
        resized_faces, face = [], None

        img = img_bgr[:, :, ::-1]
        result = self.detect_faces(img)

        if rotations is None:
            rotations = [0.]
        elif 0. not in rotations:
            rotations.insert(0, 0.)

        if len(result) != 0:
            face = max(result, key=lambda person: person["confidence"])

            if face["confidence"] >= self.alpha:
                resized_faces = [crop_and_rotate(img, self.img_shape, face["box"], angle)
                                 for angle in rotations]
                print(f"Detection time ({self.mode}): \033[1m"
                      f"{round(1000. * (timer() - start), 2)} ms\033[0m")
            else:
                print(f"{round(face['confidence'] * 100, 2)}% "
                      f"face detection confidence is too low")

        else:
            print("No face detected")

        return np.array(resized_faces), face
