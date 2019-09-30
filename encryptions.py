
"""

"encryptions.py"

AES encryption for the image database.

"""

import os
import struct
import functools

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# CONSTANTS
NEWLINE = os.linesep.encode("utf8")
_KEY_FILES = {"name": os.getenv("HOME") + "/PycharmProjects/facial-recognition/_keys/_embedding_keys.txt",
               "embedding": os.getenv("HOME") + "/PycharmProjects/facial-recognition/_keys/_name_keys.txt"}
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
def generate_cipher(key_type):
  key = get_key(key_type)
  cipher = AES.new(key, AES.MODE_EAX)
  with open(_KEY_FILES[key_type], "ab") as keys:
    keys.write(cipher.nonce)
  return cipher

# RETRIEVALS
@require_permission
def get_key(key_type):
  with open(_KEY_FILES[key_type], "rb") as keys:
    return b"".join(keys.readlines())[:_BIT_ENCRYPTION]

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
  def encrypt_data(data):
    generate_key("embedding")
    generate_key("name")

    encrypted = {}
    for person in data:
      embedding_cipher = generate_cipher("embedding")
      name_cipher = generate_cipher("name")

      encrypted_embed = list(encrypt(bytes(struct.pack("%sd" % len(data[person]), *data[person])), embedding_cipher))
      encrypted_name = "".join([chr(c) for c in list(encrypt(bytearray(person, encoding="utf8"), name_cipher))])
      # bytes are not json-serializable

      encrypted[encrypted_name] = encrypted_embed

    return encrypted

  @staticmethod
  def decrypt_data(data):
    decrypted = {}
    for nonce_pos, encrypted_name in enumerate(data):
      byte_embed = decrypt(bytes(data[encrypted_name]), key_type="embedding", position=nonce_pos)

      embed = np.array(list(struct.unpack("%sd" % (len(byte_embed) // 8), byte_embed)), dtype=np.float32)
      # using double precision (C long doubles not available in Python), hence integer division by 8 (double is 8 bits)
      name = decrypt(bytes([ord(c) for c in encrypted_name]), key_type="name", position=nonce_pos).decode("utf8")

      decrypted[name] = embed

    return decrypted
