"""AES encryption for the image database.
"""

import functools
import getpass
import struct
import sys

import dropbox
import dropbox.exceptions
import numpy as np
from Crypto.Cipher import AES  # noqa
from Crypto.Random import get_random_bytes  # noqa

sys.path.insert(1, "../")
from util.common import KEYS_ID  # noqa

NUM_BITS = 16
ALL = 0
EMBEDS = 1
NAMES = 2


def file_check(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError:
            raise OSError("permission denied (key file does not exist)")

    return _func


def maybe_load(func):
    @functools.wraps(func)
    def _func(key_file, *args, **kwargs):
        if isinstance(key_file, bytes):
            return func(key_file, *args, **kwargs)
        else:
            with open(key_file, "rb") as key:
                key = b"".join(key.readlines())
                return func(key, *args, **kwargs)

    return _func


# Generate encryption information
@file_check
def generate_key(key_file) -> None:
    open(key_file, "w").close()
    with open(key_file, "wb") as keys:
        key = get_random_bytes(NUM_BITS)
        keys.write(key)


@file_check
def generate_cipher(key_file, alloc_mem):
    key = get_key(key_file)
    cipher = AES.new(key, AES.MODE_EAX)
    if alloc_mem:
        with open(key_file, "ab") as keys:
            keys.write(cipher.nonce)
    return cipher


# RETRIEVALS
@maybe_load
def get_key(key_bytes):
    return key_bytes[:NUM_BITS]


@maybe_load
def get_nonce(key_bytes, position):
    return key_bytes[NUM_BITS:][position * NUM_BITS : (position + 1) * NUM_BITS]


# ENCRYPT AND DECRYPT
def encrypt(data, cipher):
    cipher_text, __ = cipher.encrypt_and_digest(data)
    return cipher_text


def decrypt(cipher_text, position, key_file):
    decrypt_cipher = AES.new(
        get_key(key_file), AES.MODE_EAX, nonce=get_nonce(key_file, position)
    )
    return decrypt_cipher.decrypt(cipher_text)


def encrypt_data(
    data,
    name_keys,
    embedding_keys,
    to_encrypt=ALL,
    decryptable=True,
):
    if decryptable:
        generate_key(name_keys)
        generate_key(embedding_keys)

    # Backwards compatibility with local key paths
    encrypted = {}
    for person in data:
        assert (
            data[person].ndim == 2
        ), "embeds must have shape (number of embeds, dim(embed))"
        encrypted_name, encrypted_embeds = person, data[person].tolist()

        if to_encrypt in (ALL, NAMES):
            name_cipher = generate_cipher(name_keys, alloc_mem=decryptable)
            encrypted_name = list(encrypt(person.encode("utf-8"), name_cipher))
            encrypted_name = "".join(map(chr, encrypted_name))
            # bytes are not json-serializable

        if to_encrypt in (ALL, EMBEDS):
            for idx, embed in enumerate(encrypted_embeds):
                embed_cipher = generate_cipher(embedding_keys, alloc_mem=decryptable)
                byte_embed = struct.pack("%sd" % len(embed), *embed)
                encrypted_embeds[idx] = list(encrypt(byte_embed, embed_cipher))

        encrypted[encrypted_name] = encrypted_embeds

    return encrypted


def decrypt_data(data, to_encrypt=ALL):
    """Obviously, if the embedding sets have differing lengths, the lengths
    of the nonce groups associated with those embeddings will be different.

    However, the position of the name nonce will not be affected by the
    length of the nonce group because it is stored in a different memory
    buffer than the embedding nonces. Therefore, the name nonce's position
    in its memory buffer is just the index of the name (nonce_pos).

    Embedding nonces are more complicated. We must take into account the lengths
    of the nonce groups. Therefore, the position of any specific embed (x) in
    a set of embeddings is given by the total length of all of the embeds in
    all of the embeddings prior to x (adj_nonce_pos).
    """
    # Download key file
    token = getpass.getpass("Enter token: ")
    try:
        dbx = dropbox.Dropbox(token)
        name_key_id, embed_key_id = KEYS_ID

        _, result = dbx.files_download(name_key_id)
        name_keys = result.content

        _, result = dbx.files_download(embed_key_id)
        embedding_keys = result.content
        print("[DEBUG] downloaded key files, decrypting now")

    except dropbox.exceptions.DropboxException:
        print("[DEBUG] authentification or download failed, exiting")
        sys.exit(1)

    adj_nonce_pos = 0
    decrypted = {}
    for nonce_pos, encrypted_name in enumerate(data):
        name, embeds = encrypted_name, data[encrypted_name]
        # NOTE: assume embeds have shape (number of embeds, dim(embed) * 8)
        # because double = 8 bytes

        if to_encrypt in (ALL, NAMES):
            byte_name = bytes(map(ord, encrypted_name))
            name = decrypt(byte_name, nonce_pos, name_keys).decode("utf-8")

        if to_encrypt in (ALL, EMBEDS):
            for idx, embed in enumerate(embeds):
                byte_embed = decrypt(bytes(embed), adj_nonce_pos + idx, embedding_keys)
                nums = len(byte_embed) // 8
                unpacked = struct.unpack("%sd" % nums, byte_embed)
                embeds[idx] = np.array(unpacked)
                # NOTE: using double precision, hence int division by 8 (C double is 8 bits)

        adj_nonce_pos += len(embeds)
        decrypted[name] = embeds

    return decrypted
