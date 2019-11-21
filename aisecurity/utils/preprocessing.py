"""

"aisecurity.utils.preprocessing"

Preprocessing for FaceNet.

"""

import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np


# CONSTANTS
CONSTANTS = {
    "margin": 10,
    "img_size": (None, None)
}


# IMAGE PROCESSING
def whiten(x):
    std_adj = np.maximum(np.std(x, axis=(0, 1, 2), keepdims=True), 1. / np.sqrt(x.size))
    whitened = (x - np.mean(x, axis=(0, 1, 2), keepdims=True)) / std_adj
    return whitened


def align_imgs(paths_or_imgs, margin, faces=None, checkup=False):
    if not faces and not checkup:
        detector = MTCNN()

    def align_img(path_or_img, faces, checkup):
        try:
            img = cv2.imread(path_or_img).astype(np.uint8)
        except (SystemError, TypeError):  # if img is actually image
            img = path_or_img.astype(np.uint8)

        if not checkup:
            if not faces:
                found = detector.detect_faces(img)
                assert len(found) != 0, "face was not found in {}".format(path_or_img)
                faces = found[0]["box"]

            x, y, width, height = faces
            img = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]

        resized = cv2.resize(img, CONSTANTS["img_size"])
        return resized

    return np.array([align_img(path_or_img, faces, checkup) for path_or_img in paths_or_imgs])
