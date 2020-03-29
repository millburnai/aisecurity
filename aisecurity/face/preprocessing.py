"""

"aisecurity.face.preprocessing"

Preprocessing for FaceNet.

"""

import itertools

import cv2
import numpy as np

from aisecurity.face.detection import detect_faces
from aisecurity.utils.decorators import print_time


# CONSTANTS
IMG_CONSTANTS = {
    "margin": 10,
    "img_size": (160, 160)
}


# IMAGE PROCESSING
def normalize(x, mode="per_image"):
    if mode == "per_image":
        # linearly scales x to have mean of 0, variance of 1
        std_adj = np.maximum(np.std(x, axis=(0, 1, 2), keepdims=True), 1. / np.sqrt(x.size))
        normalized = (x - np.mean(x, axis=(0, 1, 2), keepdims=True)) / std_adj
    elif mode == "fixed":
        # scales x to [-1, 1]
        normalized = (x - 127.5) / 128.0
    else:
        raise ValueError("only 'per_image' and 'fixed' standardization supported")

    return normalized


@print_time("Detection time")
def crop_face(path_or_img, margin, detector="mtcnn", alpha=0.9):
    exit_failure = itertools.repeat(-1, 2)

    try:
        img = cv2.imread(path_or_img).astype(np.uint8)[:, :, ::-1]
    except (SystemError, TypeError):  # if img is actually image
        try:
            img = path_or_img.astype(np.uint8)
        except AttributeError:
            return exit_failure

    if detector:
        result = detect_faces(img, mode=detector)
        if len(result) == 0:
            return exit_failure

        face = max(result, key=lambda person: person["confidence"])
        if face["confidence"] < alpha:
            print("{}% face detection confidence is too low".format(round(face["confidence"] * 100, 2)))
            return exit_failure

        x, y, width, height = face["box"]
        img = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]

    else:
        face = {"box": list(itertools.repeat(-1, 4)), "keypoints": {}, "confidence": 0.0}

    resized = cv2.resize(img, IMG_CONSTANTS["img_size"])
    return resized, face
