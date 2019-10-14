"""

"aisecurity.extras._data_management"

Internal use only-- manages data.

"""

from aisecurity.extras.paths import DATABASE
from aisecurity.facenet import *


def redump(json_file, ignore=None):
    data = Preprocessing.retrieve_embeds(
        os.getenv("HOME") + "/PycharmProjects/aisecurity/_local/_database/decrypted.json")
    with open(json_file, "w") as dump_file:
        json.dump(DataEncryption.encrypt_data(data, ignore=ignore), dump_file, indent=4)
    data = Preprocessing.retrieve_embeds(json_file)
    print(list(data.keys()))


def dump_from_encrypted():
    data = Preprocessing.retrieve_embeds(
        os.getenv("HOME") + "/PycharmProjects/aisecurity/_local/_database/encrypted.json")

    with open(HOME + "/database/decrypted.json", "w+") as json_file:
        data = dict((key, val.tolist()) for key, val in data.items())
        json.dump(data, json_file, indent=4)

    data = Preprocessing.retrieve_embeds(HOME + "/database/decrypted.json")
    print(data.keys())


if __name__ == "__main__":
    # redump(HOME + "/database/encrypted.json", ignore=["embeddings"])
    data = Preprocessing.retrieve_embeds(DATABASE)
    print(data.keys())
    print("Nothing for now!")