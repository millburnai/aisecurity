"""
"aisecurity.dataflow.loader"
Data loader and writer utils.
"""

from timeit import default_timer as timer
import functools
import json
import os
import warnings
import gc
import cv2
import numpy as np
import psutil

import tqdm

from aisecurity.privacy.encryptions import DataEncryption
from aisecurity.utils.paths import DATABASE_INFO, DATABASE, NAME_KEYS, EMBEDDING_KEYS


# DECORATORS
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


# LOAD ON THE FLY
@print_time("Data embedding time")
def online_load(facenet, img_dir, people=None):
    if people is None:
        people = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

    data = {}
    no_faces = []

    with tqdm.trange(len(people)) as pbar:
        for person in people:
            print(psutil.virtual_memory())
            try:
                data[person.strip(".jpg").strip(".png")] = np.squeeze(facenet.predict(cv2.imread(os.path.join(img_dir, person)), rotations=[15, 15])[0])
                pbar.update()
            except AssertionError:
                warnings.warn("face not found in {}".format(person))
                no_faces.append(person)
                continue

            gc.collect()

    return data, no_faces


# LONG TERM STORAGE
def encrypt_to_ignore(encrypt):
    ignore = ["names", "embeddings"]

    if encrypt == "all":
        encrypt = ["names", "embeddings"]

    if encrypt:
        for item in encrypt:
            ignore.remove(item)

    return ignore


@print_time("Data dumping time")
def dump_and_encrypt(data, dump_path, encrypt=None, mode="w+"):
    ignore = encrypt_to_ignore(encrypt)
    for person, embeddings in data.items():
        data[person] = [embed.tolist() for embed in embeddings]
    encrypted_data = DataEncryption.encrypt_data(data, ignore=ignore)

    with open(dump_path, mode, encoding="utf-8") as dump_file:
        json.dump(encrypted_data, dump_file, ensure_ascii=False, indent=4)

    return encrypted_data


@print_time("Data embedding and dumping time")
def dump_and_embed(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, encrypt="all", mode="w+"):
    if not full_overwrite:
        old_embeds = retrieve_embeds(retrieve_path if retrieve_path else dump_path)
        new_embeds, no_faces = online_load(facenet, img_dir)
        data = {**old_embeds, **new_embeds}
    else:
        data, no_faces = online_load(facenet, img_dir)

    encrypted_data = dump_and_encrypt(data, dump_path, encrypt=encrypt, mode=mode)
    print(dump_path)

    path_to_config = dump_path.replace(".json", "_info.json")
    with open(path_to_config, "w+", encoding="utf-8") as config_file:
        print(path_to_config)
        metadata = {"encrypted": encrypt, "metric": facenet.dist_metric.get_config()}
        json.dump(metadata, config_file, indent=4)

    return encrypted_data, no_faces


@print_time("Data retrieval time")
def retrieve_embeds(path=DATABASE, encrypted=DATABASE_INFO["encrypted"], name_keys=NAME_KEYS,
                    embedding_keys=EMBEDDING_KEYS):
    ignore = encrypt_to_ignore(encrypted)

    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return DataEncryption.decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)