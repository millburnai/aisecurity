"""

"aisecurity.dataflow.data"

Data utils.

"""

import json
import os
import warnings

import numpy as np
import tqdm

from aisecurity.privacy.encryptions import DataEncryption
from aisecurity.utils.events import print_time, in_dev
from aisecurity.utils.distance import DistMetric
from aisecurity.utils.paths import DATABASE_INFO, DATABASE, NAME_KEYS, EMBEDDING_KEYS


# LOAD ON THE FLY
@print_time("Data preprocessing time")
def online_load(facenet, img_dir, people=None):
    if people is None:
        people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]

    data = {}
    no_faces = []

    with tqdm.trange(len(people)) as pbar:
        for person in people:
            try:
                data[person.strip(".jpg").strip(".png")] = facenet.predict(os.path.join(img_dir, person))
                pbar.update()
            except AssertionError:
                warnings.warn("face not found in {}".format(person))
                no_faces.append(person)
                continue

    return data, no_faces


# LONG TERM STORAGE
@print_time("Data dumping time")
def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, to_encrypt="all", mode="w+"):
    ignore = ["names", "embeddings"]
    if to_encrypt == "all":
        to_encrypt = ["names", "embeddings"]
    for item in to_encrypt:
        ignore.remove(item)

    if not full_overwrite:
        old_embeds = retrieve_embeds(retrieve_path if retrieve_path  else dump_path)
        new_embeds, no_faces = online_load(facenet, img_dir)

        embeds_dict = {**old_embeds, **new_embeds}  # combining dicts and overwriting any duplicates with new_embeds
    else:
        embeds_dict, no_faces = online_load(facenet, img_dir)

    encrypted_data = DataEncryption.encrypt_data(embeds_dict, ignore=ignore)

    with open(dump_path, mode) as dump_file:
        json.dump(encrypted_data, dump_file, indent=4, ensure_ascii=False)

    path_to_config = dump_path.replace(".json", "_info.json")
    with open(path_to_config, "w+") as config_file:
        metadata = {"encrypted": to_encrypt, "metric": facenet.dist_metric.get_config()}
        json.dump(metadata, config_file, indent=4)

    return encrypted_data, no_faces


@print_time("Data retrieval time")
def retrieve_embeds(path=DATABASE, encrypted=DATABASE_INFO["encrypted"], name_keys=NAME_KEYS,
                    embedding_keys=EMBEDDING_KEYS):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    ignore = ["names", "embeddings"]
    if encrypted == "all":
        encrypted = ["names", "embeddings"]
    for item in encrypted:
        ignore.remove(item)

    return DataEncryption.decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)
