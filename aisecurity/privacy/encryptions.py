"""

"aisecurity.privacy.encryptions"

AES encryption for the image database.

"""

import functools
import struct

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from aisecurity.utils.paths import NAME_KEYS, EMBEDDING_KEYS


# CONSTANTS
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
def generate_key(key_file):
    open(key_file, "w", encoding="utf-8").close()
    with open(key_file, "wb", encoding="utf-8") as keys:
        key = get_random_bytes(_BIT_ENCRYPTION)
        keys.write(key)


@require_permission
def generate_cipher(key_file, alloc_mem):
    key = get_key(key_file)
    cipher = AES.new(key, AES.MODE_EAX)
    if alloc_mem:
        with open(key_file, "ab", encoding="utf-8") as keys:
            keys.write(cipher.nonce)
    return cipher


# RETRIEVALS
@require_permission
def get_key(key_file):
    with open(key_file, "rb", encoding="utf-8") as keys:
        key = b"".join(keys.readlines())[:_BIT_ENCRYPTION]
    return key


@require_permission
def get_nonce(key_file, position):
    with open(key_file, "rb", encoding="utf-8") as keys:
        joined_nonces = b"".join(keys.readlines())[_BIT_ENCRYPTION:]
        nonce = joined_nonces[position * _BIT_ENCRYPTION:(position + 1) * _BIT_ENCRYPTION]
    return nonce


# ENCRYPT AND DECRYPT
def encrypt(data, cipher):
    cipher_text, __ = cipher.encrypt_and_digest(data)
    return cipher_text


def decrypt(cipher_text, position, key_file):
    decrypt_cipher = AES.new(get_key(key_file), AES.MODE_EAX, nonce=get_nonce(key_file, position))
    return decrypt_cipher.decrypt(cipher_text)


# DATA ENCRYPTION
class DataEncryption:

    @staticmethod
    def encrypt_data(data, ignore=None, decryptable=True, name_key_file=NAME_KEYS, embeddings_key_file=EMBEDDING_KEYS):
        if ignore is None:
            ignore = []
        if decryptable:
            generate_key(name_key_file)
            generate_key(embeddings_key_file)

        encrypted = {}
        for person in data:
            name_cipher = generate_cipher(name_key_file, alloc_mem=decryptable)
            embedding_cipher = generate_cipher(embeddings_key_file, alloc_mem=decryptable)

            encrypted_name, encrypted_embed = person, data[person]

            if isinstance(encrypted_embed, np.ndarray):
                encrypted_embed = encrypted_embed.reshape(-1,).tolist()

            if "names" not in ignore:
                encrypted_name = [chr(c) for c in list(encrypt(person.encode("utf-8"), name_cipher))]
                encrypted_name = "".join(encrypted_name)
                # bytes are not json-serializable
            if "embeddings" not in ignore:
                encrypted_embed = bytes(struct.pack("%sd" % len(data[person]), *data[person]))
                encrypted_embed = list(encrypt(encrypted_embed, embedding_cipher))

            encrypted[encrypted_name] = encrypted_embed

        return encrypted

    @staticmethod
    def decrypt_data(data, ignore=None, name_keys=NAME_KEYS, embedding_keys=EMBEDDING_KEYS):
        if ignore is None:
            ignore = []

        decrypted = {}
        for nonce_pos, encrypted_name in enumerate(data):
            name, embed = encrypted_name, data[encrypted_name]
            if "names" not in ignore:
                name = decrypt(bytes([ord(c) for c in encrypted_name]), nonce_pos, name_keys)
                name = name.decode("utf-8")
            if "embeddings" not in ignore:
                byte_embed = decrypt(bytes(data[encrypted_name]), nonce_pos, embedding_keys)
                embed = np.array(list(struct.unpack("%sd" % (len(byte_embed) // 8), byte_embed)), dtype=np.float32)
                # using double precision (C long doubles not available), hence int division by 8 (double is 8 bits)

            decrypted[name] = embed

        return decrypted
