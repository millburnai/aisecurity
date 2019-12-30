"""

"aisecurity.data.dataflow"

Data utils.

"""

import json
import os
import tqdm
import warnings
from mtcnn.mtcnn import MTCNN
import cv2

from aisecurity.privacy.encryptions import DataEncryption
from aisecurity.utils.misc import timer, isolate_face
from aisecurity.utils.paths import CONFIG_HOME


# LOAD ON THE FLY
@timer(message="Data preprocessing time")
def online_load(facenet, img_dir, people=None):
    mtcnn = MTCNN()
    face_cascade = cv2.CascadeClassifier(CONFIG_HOME + '/models/haarcascade_frontalface_default.xml')
    if people is None:
        people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]

    data = {}
    no_faces = []
    with tqdm.trange(len(people)) as pbar:
        for person in people:
            try:
                data[person.strip(".jpg").strip(".png")] = facenet.predict([os.path.join(img_dir, person)])
                pbar.update()
            except AssertionError:
                warnings.warn("face not found in {}".format(person))
                no_faces.append(person)
                continue

    return data, no_faces


# LONG TERM STORAGE
@timer(message="Data dumping time")
def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, ignore_encrypt=None,
                retrieve_encryption=None, mode="a+"):
    if ignore_encrypt == "all":
        ignore_encrypt = ["names", "embeddings"]
    elif ignore_encrypt is not None:
        ignore_encrypt = [ignore_encrypt]

    if not full_overwrite:
        old_embeds = retrieve_embeds(retrieve_path if retrieve_path is not None else dump_path,
                                     encrypted=retrieve_encryption)
        new_embeds, no_faces = online_load(facenet, img_dir)

        embeds_dict = {**old_embeds, **new_embeds}  # combining dicts and overwriting any duplicates with new_embeds
    else:
        embeds_dict, no_faces = online_load(facenet, img_dir)

    encrypted_data = DataEncryption.encrypt_data(embeds_dict, ignore=ignore_encrypt)

    with open(dump_path, mode) as json_file:
        json.dump(encrypted_data, json_file, indent=4, ensure_ascii=False)

    return encrypted_data, no_faces


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
