"""Data loader and writer utils.

"""

import functools
import json
import ntpath
import os
from timeit import default_timer as timer
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "../")
from privacy.encryptions import (ALL, NAMES, EMBEDS,  # noqa
                                 encrypt_data, decrypt_data)
from util.paths import DB_LOB, NAME_KEY_PATH, EMBED_KEY_PATH  # noqa


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def print_time(message):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            print(f"[DEBUG] {message}: {round(timer() - start, 4)}s")
            return result
        return _func
    return _timer


def get_frozen_graph(path):
    """Gets frozen graph from .pb file (TF only)
    :param path: path to .pb frozen graph file
    :returns: tf.GraphDef object
    """
    import tensorflow.compat.v1 as tf  # noqa
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


def strip_id(name, split="-"):
    idx = name.rfind(split)
    return name[:idx] if idx != -1 else name


@print_time("data embedding time")
def online_load(facenet, img_dir, people=None, **kwargs):
    if people is None:
        people = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.endswith("jpg") or f.endswith("png")]

    data = {}
    no_faces = []

    for person in tqdm(people):
        if not person.endswith("jpg") and not person.endswith("png"):
            print(f"[DEBUG] '{person}' not a jpg or png image")
        elif os.path.getsize(person) > 1e8:
            print(f"[DEBUG] '{person}' too large (> 100M bytes)")
        else:
            no_faces.append(person)

            try:
                embeds, __ = facenet.predict(cv2.imread(person), **kwargs)

                person = ntpath.basename(person)
                person = person.replace(".jpg", "").replace(".png", "")
                data[person] = embeds.reshape(len(embeds), -1)
            except AssertionError as e:
                print(f"[DEBUG] error ('{person}'): {e}")

    return data, no_faces


def dump_and_encrypt(data, metadata, dump_path, to_encrypt=ALL,
                     mode="w+", **kwargs):
    encrypted_data = encrypt_data(data, to_encrypt=to_encrypt, **kwargs)

    with open(dump_path, mode, encoding="utf-8") as dump_file:
        data = {"metadata": metadata, "data": encrypted_data}
        json.dump(data, dump_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


@print_time("data embedding and dumping time")
def dump_and_embed(facenet, img_dir, dump_path, retrieve_path=None,
                   full_overwrite=False, to_encrypt=ALL, use_mean=False,
                   **kwargs):
    metadata = facenet.metadata
    metadata["to_encrypt"] = to_encrypt

    if not full_overwrite:
        path = retrieve_path if retrieve_path else dump_path
        old_embeds, old_metadata = retrieve_embeds(path, NAME_KEY_PATH,
                                                   EMBED_KEY_PATH)

        new_embeds, no_faces = online_load(facenet, img_dir, **kwargs)
        data = {**old_embeds, **new_embeds}

        assert not old_metadata or metadata == old_metadata, \
            "metadata inconsistent"

    else:
        data, no_faces = online_load(facenet, img_dir, **kwargs)

    if use_mean:
        embeds = np.array(list(data.values()))
        metadata["mean"] = np.average(embeds, axis=(0, 1, 2))

    dump_and_encrypt(data, metadata, dump_path, to_encrypt=to_encrypt)


@print_time("data retrieval time")
def retrieve_embeds(path, name_keys, embedding_keys):
    if path is None or name_keys is None or embedding_keys is None:
        return {}, {}

    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        metadata = data["metadata"]
        encrypted_data = data["data"]

    decrypted = decrypt_data(encrypted_data, metadata["to_encrypt"],
                             name_keys, embedding_keys)
    return decrypted, metadata
