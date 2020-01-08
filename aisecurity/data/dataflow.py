"""

"aisecurity.data.dataflow"

Data utils.

"""

import json
import os
import tqdm
import warnings

from aisecurity.privacy.encryptions import DataEncryption, _KEY_FILES
from aisecurity.utils.events import timer
from aisecurity.utils.paths import DATABASE_INFO


# LOAD ON THE FLY
@timer(message="Data preprocessing time")
def online_load(facenet, img_dir, people=None):
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
def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, to_encrypt="all", mode="a+"):
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

    with open(dump_path, mode) as json_file:
        json.dump(encrypted_data, json_file, indent=4, ensure_ascii=False)

    path_to_config = dump_path.replace(".json", "_info.json")
    with open(path_to_config, "w+") as json_file:
        json.dump({"encrypted": to_encrypt, "metric":"euclidean+l2_normalize"}, json_file, indent=4, ensure_ascii=False)

    return encrypted_data, no_faces


@timer(message="Data retrieval time")
def retrieve_embeds(path, encrypted=DATABASE_INFO["encrypted"], name_keys=_KEY_FILES["names"],
                    embedding_keys=_KEY_FILES["embeddings"]):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    ignore = ["names", "embeddings"]
    if encrypted == "all":
        encrypted = ["names", "embeddings"]
    for item in encrypted:
        ignore.remove(item)

    return DataEncryption.decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)


if __name__ == "__main__":
    import editdistance

    data = retrieve_embeds("/home/ryan/.aisecurity/database/embeddings.json", encrypted=["names"],
                           name_keys="/home/ryan/.aisecurity/keys/name_keys.txt",
                           embedding_keys="/home/ryan/.aisecurity/keys/embedding_keys.txt")

    with open("/home/ryan/scratchpad/aisecurity/people.txt", "r") as file:
        aisecurity_students = file.readlines()

    filtered_data = {}
    for student in aisecurity_students:
        closest_match, _ = min(
            [(person, editdistance.eval(student, person)) for person in data.keys()],
            key=lambda t: t[1]
        )
        filtered_data[student.strip("\n")] = data[closest_match]

    filtered_data["ryan_park"] = data["ryan_park"]
    filtered_data["liam_pilarski"] = data["liam_pilarski"]

    encrypted_data = DataEncryption.encrypt_data(filtered_data, ignore=["embeddings"])
    with open("/home/ryan/.aisecurity/database/test.json", "w") as file:
        json.dump(encrypted_data, file, indent=4, ensure_ascii=False)
