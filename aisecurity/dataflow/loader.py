"""

"aisecurity.dataflow.loader"

Data loader and writer utils.

"""

import functools
import json
import os
from timeit import default_timer as timer
import warnings

import cv2
import tqdm

from aisecurity.privacy.encryptions import encrypt_data, decrypt_data
from aisecurity.utils.paths import db_info, db_loc, name_key_path, embed_key_path


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
def online_load(facenet, img_dir, people=None, **kwargs):
    if people is None:
        people = [f for f in os.listdir(img_dir) if f.endswith("jpg") or f.endswith("png")]

    data = {}
    no_faces = []

    with tqdm.trange(len(people)) as pbar:
        for person in people:
            try:
                assert person.endswith("jpg") or person.endswith("png") and os.path.getsize(person) < 1e6

                img = cv2.imread(os.path.join(img_dir, person))
                print(kwargs)
                embeds, __ = facenet.predict(img, **kwargs)
                data[person.strip("jpg").strip("png")] = embeds.reshape(len(embeds), -1)

            except AssertionError:
                warnings.warn("face not found or file too large ({})".format(person))
                no_faces.append(person)
                continue

        pbar.update()

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
def dump_and_encrypt(data, dump_path, encrypt=None, mode="w+", **kwargs):
    ignore = encrypt_to_ignore(encrypt)
    encrypted_data = encrypt_data(data, ignore=ignore, **kwargs)

    with open(dump_path, mode, encoding="utf-8") as dump_file:
        json.dump(encrypted_data, dump_file, ensure_ascii=False, indent=4)

    return encrypted_data


@print_time("Data embedding and dumping time")
def dump_and_embed(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, encrypt="all", **kwargs):
    if not full_overwrite:
        old_embeds = retrieve_embeds(retrieve_path if retrieve_path else dump_path)
        new_embeds, no_faces = online_load(facenet, img_dir, **kwargs)
        data = {**old_embeds, **new_embeds}
    else:
        data, no_faces = online_load(facenet, img_dir, **kwargs)

    encrypted_data = dump_and_encrypt(data, dump_path, encrypt=encrypt, **kwargs)

    path_to_config = dump_path.replace(".json", "_info.json")
    with open(path_to_config, "w+", encoding="utf-8") as config_file:
        metadata = {"encrypted": encrypt, "metric": facenet.dist_metric.get_config()}
        json.dump(metadata, config_file, indent=4)

    return encrypted_data, no_faces


@print_time("Data retrieval time")
def retrieve_embeds(path=db_loc, encrypted=db_info["encrypted"], name_keys=name_key_path,
                    embedding_keys=embed_key_path):
    ignore = encrypt_to_ignore(encrypted)

    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)
