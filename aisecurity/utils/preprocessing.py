"""

"aisecurity.utils.preprocessing"

Preprocessing for FaceNet.

"""

import functools
import time

import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np


# CONSTANTS
CONSTANTS = {
    "margin": 10,
    "img_size": (None, None)
}


# DECORATORS
def timer(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(time.time() - start, 3)))
            return result

        return _func

    return _timer


# IMAGE PROCESSING
def whiten(x):
    std_adj = np.maximum(np.std(x, axis=(0, 1, 2), keepdims=True), 1. / np.sqrt(x.size))
    whitened = (x - np.mean(x, axis=(0, 1, 2), keepdims=True)) / std_adj
    return whitened


def align_imgs(paths_or_imgs, margin, faces=None):
    if not faces:
        detector = MTCNN()

    def align_img(path_or_img, faces=None):
        try:
            img = cv2.imread(path_or_img).astype(np.uint8)
        except SystemError:  # if img is actually image
            img = path_or_img.astype(np.uint8)

        # print(img.shape)
        # height, width, channels = img.shape
        # scale_factor = CONSTANTS["img_size"][0] / ((height + width) / 2)
        # img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        #
        # plt.imshow(img); plt.show()
        # print(img.shape)

        if not faces:
            found = detector.detect_faces(img)
            assert len(found) != 0, "face was not found in {}".format(path_or_img)
            faces = found[0]["box"]

        x, y, width, height = faces
        cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
        resized = cv2.resize(cropped, CONSTANTS["img_size"])
        return resized

    return np.array([align_img(path_or_img, faces=faces) for path_or_img in paths_or_imgs])
