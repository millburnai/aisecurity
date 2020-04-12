"""

"aisecurity.face.preprocessing"

Preprocessing for FaceNet.

"""

import cv2
import numpy as np

from aisecurity.dataflow.loader import print_time
from aisecurity.face.detection import detect_faces


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
def crop_face(img, margin, detector="mtcnn", alpha=0.9, rotations=None):
    def crop_and_rotate(img, face_coords, rotation_angle):
        x, y, width, height = face_coords
        img = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]

        resized = cv2.resize(img, IMG_CONSTANTS["img_size"])
        if rotation_angle == 0:
            return resized
        elif rotation_angle == -1:
            return cv2.flip(resized, 1)
        else:
            # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
            rotation_matrix = cv2.getRotationMatrix2D(tuple(np.array(resized.shape[1::-1]) / 2), rotation_angle, 1.)
            return cv2.warpAffine(resized, rotation_matrix, resized.shape[1::-1], flags=cv2.INTER_LINEAR)


    resized_faces, face = [], None

    if rotations is None:
        rotations = [0.]
    elif 0. not in rotations:
        rotations.append(0.)

    if detector:
        result = detect_faces(img, mode=detector)

        if len(result) != 0:
            face = max(result, key=lambda person: person["confidence"])

            if face["confidence"] >= alpha:
                resized_faces = [crop_and_rotate(img, face["box"], angle) for angle in sorted(rotations)]
            else:
                print("{}% face detection confidence is too low".format(round(face["confidence"] * 100, 2)))

    return resized_faces, face
