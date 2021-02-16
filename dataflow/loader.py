"""Data loader and writer utils.

"""

import functools
import json
import os
from timeit import default_timer as timer
import sys

import cv2
import numpy as np
import tensorflow.compat.v1 as tf  # noqa
from tqdm import tqdm

sys.path.insert(1, "../")
from privacy.encryptions import encrypt_data, decrypt_data  # noqa
from utils.paths import db_loc, name_key_path, embed_key_path  # noqa


def print_time(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(timer() - start, 4)))
            return result

        return _func

    return _timer


def get_frozen_graph(path):
    """Gets frozen graph from .pb file (TF only)
    :param path: path to .pb frozen graph file
    :returns: tf.GraphDef object
    """

    with tf.gfile.FastGFile(path, "rb") as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
    return graph_def


def screen_data(key, value):
    """Checks if key-value pair is valid for data dict
    :param key: new key
    :param value: new value
    """

    assert isinstance(key, str), "data keys must be person names"

    for embed in value:
        embed = np.asarray(embed)
        is_vector = np.prod(embed.shape) == embed.flatten().shape
        assert is_vector, f"each value must be a vectorized " \
                          f"embedding, got shape {embed.shape}"

    return key, value


@print_time("Data embedding time")
def online_load(facenet, img_dir, people=None, **kwargs):
    if people is None:
        people = [f for f in os.listdir(img_dir)
                  if f.endswith("jpg") or f.endswith("png")]

    data = {}
    no_faces = []

    for person in tqdm(people):
        if not person.endswith("jpg") and not person.endswith("png"):
            print(f"[DEBUG] '{person}' not a jpg or png image")
        elif os.path.getsize(person) < 1e6:
            print(f"[DEBUG] '{person}' too large (> 1M bytes)")
            no_faces.append(person)

            img = cv2.imread(os.path.join(img_dir, person))
            embeds, __ = facenet.predict(img, **kwargs)

            person = person.strip("jpg").strip("png")
            data[person] = embeds.reshape(len(embeds), -1)

    return data, no_faces


def encrypt_to_ignore(encrypt):
    ignore = ["names", "embeddings"]
    if encrypt == "all":
        encrypt = ["names", "embeddings"]
    if encrypt:
        for item in encrypt:
            ignore.remove(item)
    return ignore


def dump_and_encrypt(data, metadata, dump_path, encrypt=None,
                     mode="w+", **kwargs):
    ignore = encrypt_to_ignore(encrypt)
    encrypted_data = encrypt_data(data, ignore=ignore, **kwargs)

    with open(dump_path, mode, encoding="utf-8") as dump_file:
        data = {"metadata": metadata, "data": encrypted_data}
        json.dump(data, dump_file, ensure_ascii=False, indent=4)


@print_time("Data embedding and dumping time")
def dump_and_embed(facenet, img_dir, dump_path, retrieve_path=None,
                   full_overwrite=False, encrypt="all", **kwargs):
    metadata = facenet.metadata
    metadata["encrypt"] = encrypt

    if not full_overwrite:
        path = retrieve_path if retrieve_path else dump_path
        old_embeds, old_metadata = retrieve_embeds(path)

        new_embeds, no_faces = online_load(facenet, img_dir, **kwargs)
        data = {**old_embeds, **new_embeds}

        assert metadata == old_metadata, "metadata inconsistent"

    else:
        data, no_faces = online_load(facenet, img_dir, **kwargs)

    dump_and_encrypt(data, metadata, dump_path, encrypt=encrypt)


@print_time("Data retrieval time")
def retrieve_embeds(path, name_keys, embedding_keys):
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        metadata = data["metadata"]
        encrypted_data = data["data"]

    decrypted = decrypt_data(encrypted_data, metadata["ignore"],
                             name_keys, embedding_keys)
    return decrypted, metadata
