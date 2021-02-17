"""Haarcascade or MTCNN face detection.
"""

import sys
from timeit import default_timer as timer

import cv2
from termcolor import colored

sys.path.insert(1, "../")
from util.paths import CONFIG_HOME


class FaceDetector:
    MODES = ["mtcnn", "haarcascade", "trt-mtcnn"]

    def __init__(self, mode, img_shape=(160, 160), alpha=0.8, **kwargs):
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
            from mtcnn import MTCNN
            self.mtcnn = MTCNN(min_face_size=self.kwargs["min_face_size"])

        if "haarcascade" in mode:
            hpath = CONFIG_HOME + "models/haarcascade_frontalface_default.xml"
            self.haarcascade = cv2.CascadeClassifier(hpath)

    def detect_faces(self, img):
        result = []

        if "trt-mtcnn" in self.mode:
            result = self.trt_mtcnn.detect_faces(img)

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

    def crop_face(self, img_bgr, margin, flip=False, verbose=True):
        start = timer()
        resized_faces, face = None, None

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.detect_faces(img)

        if len(result) != 0:
            face = max(result, key=lambda person: person["confidence"])

            if face["confidence"] >= self.alpha:
                x, y, width, height = face["box"]
                img = img[y - margin // 2:y + height + margin // 2,
                          x - margin // 2:x + width + margin // 2, :]
                resized_faces = [cv2.resize(img, self.img_shape)]

                if flip:
                    flipped = cv2.flip(resized_faces[0], 1)
                    resized_faces.append(flipped)

                if verbose:
                    elapsed = round(1000. * (timer() - start), 2)
                    time = colored(f"{elapsed} ms", attrs=["bold"])
                    print(f"Detection time ({self.mode}): " + time)

            elif verbose:
                confidence = round(face['confidence'] * 100, 2)
                print(f"{confidence}% face detection confidence is too low")

        elif verbose:
            print("No face detected")

        return resized_faces, face
