"""

"aisecurity.encryptions"

AES encryption for the image database.

"""

import functools
import json
import os
import struct
import subprocess

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from aisecurity.extras.paths import KEY_DIR, KEY_FILE

# CONSTANTS
try:
    _KEY_FILES = json.load(open(KEY_FILE))
    assert os.path.exists(_KEY_FILES["names"]) and os.path.exists(_KEY_FILES["embeddings"])
except (FileNotFoundError, AssertionError):
    subprocess.call(["make_keys.sh", KEY_DIR, KEY_FILE])
    _KEY_FILES = json.load(open(KEY_FILE))

_BIT_ENCRYPTION = 16


# DECORATORS
def require_permission(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError:
            raise OSError("permission denied (key file does not exist)")

    return _func


# GENERATING ENCRYPTION INFO
@require_permission
def generate_key(key_type):
    open(_KEY_FILES[key_type], "w").close()
    with open(_KEY_FILES[key_type], "wb") as keys:
        key = get_random_bytes(_BIT_ENCRYPTION)
        keys.write(key)


@require_permission
def generate_cipher(key_type, alloc_mem):
    key = get_key(key_type)
    cipher = AES.new(key, AES.MODE_EAX)
    if alloc_mem:
        with open(_KEY_FILES[key_type], "ab") as keys:
            keys.write(cipher.nonce)
    return cipher


# RETRIEVALS
@require_permission
def get_key(key_type):
    with open(_KEY_FILES[key_type], "rb") as keys:
        key = b"".join(keys.readlines())[:_BIT_ENCRYPTION]
    return key


@require_permission
def get_nonce(key_type, position):
    with open(_KEY_FILES[key_type], "rb") as keys:
        joined_nonces = b"".join(keys.readlines())[_BIT_ENCRYPTION:]
        nonce = joined_nonces[position * _BIT_ENCRYPTION:(position + 1) * _BIT_ENCRYPTION]
    return nonce


# ENCRYPT AND DECRYPT
def encrypt(data, cipher):
    cipher_text, tag = cipher.encrypt_and_digest(data)
    return cipher_text


def decrypt(cipher_text, key_type, position):
    decrypt_cipher = AES.new(get_key(key_type), AES.MODE_EAX, nonce=get_nonce(key_type, position))
    return decrypt_cipher.decrypt(cipher_text)


# DATA ENCRYPTION
class DataEncryption(object):

    @staticmethod
    def encrypt_data(data, ignore=None, decryptable=True):
        if ignore is None:
            ignore = []
        if decryptable:
            generate_key("names")
            generate_key("embeddings")

        encrypted = {}
        for person in data:
            name_cipher = generate_cipher("names", alloc_mem=decryptable)
            embedding_cipher = generate_cipher("embeddings", alloc_mem=decryptable)

            encrypted_name, encrypted_embed = person, data[person]
            if "names" not in ignore:
                encrypted_name = [chr(c) for c in list(encrypt(bytearray(person, encoding="utf8"), name_cipher))]
                encrypted_name = "".join(encrypted_name)
                # bytes are not json-serializable
            if "embeddings" not in ignore:
                encrypted_embed = bytes(struct.pack("%sd" % len(data[person]), *data[person]))
                encrypted_embed = list(encrypt(encrypted_embed, embedding_cipher))

            encrypted[encrypted_name] = encrypted_embed

        return encrypted

    @staticmethod
    def decrypt_data(data, ignore=None):
        if ignore is None:
            ignore = []

        decrypted = {}
        for nonce_pos, encrypted_name in enumerate(data):
            name, embed = encrypted_name, data[encrypted_name]
            if "names" not in ignore:
                name = decrypt(bytes([ord(c) for c in encrypted_name]), key_type="names", position=nonce_pos)
                name = name.decode("utf8")
            if "embeddings" not in ignore:
                byte_embed = decrypt(bytes(data[encrypted_name]), key_type="embeddings", position=nonce_pos)
                embed = np.array(list(struct.unpack("%sd" % (len(byte_embed) // 8), byte_embed)), dtype=np.float32)
                # using double precision (C long doubles not available), hence int division by 8 (double is 8 bits)

            decrypted[name] = embed

        return decrypted
