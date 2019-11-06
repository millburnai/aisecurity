
"""

"aisecurity.preprocessing"

Preprocessing and data handling for FaceNet.

"""

import functools
import json
import os
import time

import cv2
import numpy as np
from imageio import imread
from mtcnn.mtcnn import MTCNN

from aisecurity.encryptions import DataEncryption


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


# BASE FUNCTIONS
def whiten(x):
    std_adj = np.maximum(np.std(x, axis=(0, 1, 2), keepdims=True), 1. / np.sqrt(x.size))
    whitened = (x - np.mean(x, axis=(0, 1, 2), keepdims=True)) / std_adj
    return whitened


def align_imgs(paths_or_imgs, margin, faces=None):
    if not faces:
        detector = MTCNN()

    def align_img(path_or_img, faces=None):
        try:
            img = imread(path_or_img)
        except OSError:  # if img is embedding
            img = path_or_img

        if not faces:
            found = detector.detect_faces(img)
            assert len(found) != 0, "face was not found in {}".format(path_or_img)
            faces = found[0]["box"]

        x, y, width, height = faces
        cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
        resized = cv2.resize(cropped, CONSTANTS["img_size"])
        return resized

    return np.array([align_img(path_or_img, faces=faces) for path_or_img in paths_or_imgs])


def embed(sess, paths_or_imgs, input_name, output_tensor, margin=CONSTANTS["margin"], faces=None):
    l2_normalize = lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-6))

    aligned_imgs = whiten(align_imgs(paths_or_imgs, margin, faces=faces))
    raw_embeddings = sess.run(output_tensor, {input_name: aligned_imgs})
    normalized_embeddings = l2_normalize(raw_embeddings)

    return normalized_embeddings


# LOADING
@timer(message="Data preprocessing time")
def load(facenet, img_dir, people=None):
    if people is None:
        people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
    data = {person: facenet.predict(img_dir + person) for person in people}
    return data


@timer(message="Data dumping time")
def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, ignore_encrypt=None):

    if ignore_encrypt == "all":
        ignore_encrypt = ["names", "embeddings"]
    elif ignore_encrypt is not None:
        ignore_encrypt = [ignore_encrypt]

    if not full_overwrite:
        people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
        old_embeds = retrieve_embeds(retrieve_path if retrieve_path is not None else dump_path)

        new_people = [person for person in people if person not in old_embeds.keys()]
        new_embeds = load(facenet, img_dir, people=new_people)

        embeds_dict = {**old_embeds, **new_embeds}  # combining dicts and overwriting any duplicates with new_embeds
    else:
        embeds_dict = load(facenet, img_dir)

    encrypted_data = DataEncryption.encrypt_data(embeds_dict, ignore=ignore_encrypt)

    with open(dump_path, "w+") as json_file:
        json.dump(encrypted_data, json_file, indent=4, ensure_ascii=False)


@timer(message="Data retrieval time")
def retrieve_embeds(path, encrypted=None):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    if encrypted == "embeddings":
        return DataEncryption.decrypt_data(data, ignore=["names"])
    elif encrypted == "names":
        return DataEncryption.decrypt_data(data, ignore=["embeddings"])
    elif encrypted == "all":
        return DataEncryption.decrypt_data(data, ignore=None)
    else:
        return data
