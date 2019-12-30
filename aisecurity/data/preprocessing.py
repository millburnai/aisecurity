"""

"aisecurity.data.preprocessing"

Preprocessing for FaceNet.

"""
from aisecurity.utils.paths import CONFIG_HOME

import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np


# CONSTANTS
IMG_CONSTANTS = {
    "margin": 10,
    "img_size": (None, None),
    "mtcnn": MTCNN(),
    "face_cascade": cv2.CascadeClassifier(CONFIG_HOME + '/models/haarcascade_frontalface_default.xml')
}


# IMAGE PROCESSING
def normalize(x):
    std_adj = np.maximum(np.std(x, axis=(0, 1, 2), keepdims=True), 1. / np.sqrt(x.size))
    normalized = (x - np.mean(x, axis=(0, 1, 2), keepdims=True)) / std_adj
    return normalized


def align_imgs(paths_or_imgs, margin, faces=None, checkup=False):

    def align_img(path_or_img, faces, checkup):
        try:
            img = cv2.imread(path_or_img).astype(np.uint8)
        except (SystemError, TypeError):  # if img is actually image
            img = path_or_img.astype(np.uint8)

        if not checkup:
            if not faces:
                found = IMG_CONSTANTS["mtcnn"].detect_faces(img)
                if len(found) != 0:
                    faces = found[0]["box"]
                else:
                    found = IMG_CONSTANTS["face_cascade"].detectMultiScale(img, scaleFactor=1.1)
                    assert len(found) != 0, "face was not found in {}".format(path_or_img)
                    faces = found[0]

            x, y, width, height = faces
            img = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]

        resized = cv2.resize(img, IMG_CONSTANTS["img_size"])
        return resized

    return np.array([align_img(path_or_img, faces, checkup) for path_or_img in paths_or_imgs])
