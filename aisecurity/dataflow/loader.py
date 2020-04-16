"""

"aisecurity.dataflow.loader"

Data loader and writer utils.

"""

from timeit import default_timer as timer
import functools
import json
import os
import warnings

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

    with open(dump_path, mode, encoding="utf-8") as dump_file:
        json.dump(encrypted_data, dump_file, ensure_ascii=False, indent=4)

    return encrypted_data



@print_time("Data retrieval time")
def retrieve_embeds(path=DATABASE, encrypted=DATABASE_INFO["encrypted"], name_keys=NAME_KEYS,
                    embedding_keys=EMBEDDING_KEYS):
    ignore = encrypt_to_ignore(encrypted)

    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return DataEncryption.decrypt_data(data, ignore=ignore, name_keys=name_keys, embedding_keys=embedding_keys)
