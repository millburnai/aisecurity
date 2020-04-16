"""

"aisecurity.face.preprocessing"

Preprocessing for FaceNet.

"""

from timeit import default_timer as timer

import cv2
import numpy as np

from aisecurity.face.detection import detect_faces


# GLOBALS
IMG_SHAPE = (160, 160)


def set_img_shape(img_shape):
    global IMG_SHAPE

    IMG_SHAPE = tuple(img_shape)


# IMAGE PROCESSING
def normalize(imgs, mode="per_image"):
    if mode == "per_image":
        # linearly scales x to have mean of 0, variance of 1
        std_adj = np.maximum(np.std(imgs, axis=(1, 2, 3), keepdims=True), 1. / np.sqrt(imgs.size / len(imgs)))
        normalized = (imgs - np.mean(imgs, axis=(1, 2, 3), keepdims=True)) / std_adj
    elif mode == "fixed":
        # scales x to [-1, 1]
        normalized = (imgs - 127.5) / 128.
    else:
        raise ValueError("only 'per_image' and 'fixed' standardization supported")

    return normalized


def crop_face(img, margin, detector="mtcnn", alpha=0.9, rotations=None):
    def crop_and_rotate(img, face_coords, rotation_angle):
        x, y, width, height = face_coords
        img = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]

        resized = cv2.resize(img, IMG_SHAPE)
        if rotation_angle == 0:
            return resized
        elif rotation_angle == -1:
            return cv2.flip(resized, 1)
        else:
            # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
            rotation_matrix = cv2.getRotationMatrix2D(tuple(np.array(resized.shape[1::-1]) / 2), rotation_angle, 1.)
            return cv2.warpAffine(resized, rotation_matrix, resized.shape[1::-1], flags=cv2.INTER_LINEAR)

    start = timer()
    resized_faces, face = [], None

    if rotations is None:
        rotations = [0.]
    elif 0. not in rotations:
        rotations.append(0.)

    if detector:
        result = detect_faces(img, mode=detector, alpha=alpha)

        if len(result) != 0:
            face = max(result, key=lambda person: person["confidence"])

            if face["confidence"] >= alpha:
                resized_faces = [crop_and_rotate(img, face["box"], angle) for angle in sorted(rotations)]
                print("Detection time ({}): \033[1m{} ms\033[0m".format(detector, round(1000. * (timer() - start), 2)))
            else:
                print("{}% face detection confidence is too low".format(round(face["confidence"] * 100, 2)))

        else:
            print("No face detected")

    return np.array(resized_faces), face
