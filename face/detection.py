"""Haarcascade or MTCNN face detection.
"""

import sys
from timeit import default_timer as timer

import cv2
from mtcnn import MTCNN
import numpy as np

sys.path.insert(1, "../")
from utils.paths import config_home


class FaceDetector:
    MODES = ["mtcnn", "haarcascade", "trt-mtcnn"]

    def __init__(self, mode, img_shape=(160, 160), alpha=0.9, **kwargs):
        assert any(det in mode for det in self.MODES), \
            "supported modes are 'mtcnn', 'haarcascade', and 'trt-mtcnn'"

        self.mode = mode
        self.alpha = alpha
        self.img_shape = tuple(img_shape)
        self.kwargs = kwargs

        if "min_face_size" not in self.kwargs:
            self.kwargs["min_face_size"] = 20

        if "trt-mtcnn" in mode:
            sys.path.insert(1, "../face/trt_mtcnn_plugin")
            from face.trt_mtcnn_plugin.trt_mtcnn import TrtMTCNNWrapper

            engine_paths = [f"../face/trt_mtcnn_plugin/mtcnn/det{i+1}.engine"
                            for i in range(3)]
            self.trt_mtcnn = TrtMTCNNWrapper(*engine_paths)

        if "mtcnn" in mode.replace("trt-mtcnn", ""):
            self.mtcnn = MTCNN(min_face_size=self.kwargs["min_face_size"])

        if "haarcascade" in mode:
            hpath = config_home + "models/haarcascade_frontalface_default.xml"
            self.haarcascade = cv2.CascadeClassifier(hpath)

    def detect_faces(self, img):
        result = []

        if "trt-mtcnn" in self.mode:
            minsize = max(40, int(self.kwargs["min_face_size"]))
            result = self.trt_mtcnn.detect_faces(img, minsize=minsize)

        if "mtcnn" in self.mode.replace("trt-mtcnn", ""):
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

    def crop_face(self, img_bgr, margin, rotations=None, verbose=True):
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
                if verbose:
                    print(f"Detection time ({self.mode}): \033[1m"
                          f"{round(1000. * (timer() - start), 2)} ms\033[0m")
            elif verbose:
                print(f"{round(face['confidence'] * 100, 2)}% "
                      f"face detection confidence is too low")

        elif verbose:
            print("No face detected")

        return np.array(resized_faces), face
