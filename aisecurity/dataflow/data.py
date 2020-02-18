"""

"aisecurity.dataflow.data"

Data utils.

"""

import json
import os
import warnings

import tqdm

from aisecurity.privacy.encryptions import DataEncryption
from aisecurity.utils.events import print_time
from aisecurity.utils.paths import DATABASE_INFO, DATABASE, NAME_KEYS, EMBEDDING_KEYS


# LOAD ON THE FLY
@print_time("Data embedding time")
def online_load(facenet, img_dir, people=None):
    if people is None:
        people = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

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
def encrypt_to_ignore(encrypt):
    ignore = ["names", "embeddings"]
    if encrypt == "all":
        encrypt = ["names", "embeddings"]
    for item in encrypt:
        ignore.remove(item)

    return ignore


@print_time("Data dumping time")
def dump_and_encrypt(data, dump_path, encrypt=None, mode="w+"):
    ignore = encrypt_to_ignore(encrypt)
    for person, embeddings in data.items():
        data[person] = [embed.tolist() for embed in embeddings]
    encrypted_data = DataEncryption.encrypt_data(data, ignore=ignore)

    with open(dump_path, mode) as dump_file:
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

    path_to_config = dump_path.replace(".json", "_info.json")
    with open(path_to_config, "w+") as config_file:
        metadata = {"encrypted": encrypt, "metric": facenet.dist_metric.get_config()}
        json.dump(metadata, config_file, indent=4)

    return encrypted_data, no_faces


@print_time("Data retrieval time")
def retrieve_embeds(path=DATABASE, encrypted=DATABASE_INFO["encrypted"], name_keys=NAME_KEYS,
                    embedding_keys=EMBEDDING_KEYS):
    ignore = encrypt_to_ignore(encrypted)

    with open(path, "r") as json_file:
        data = json.load(json_file)

    return DataEncryption.decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)
