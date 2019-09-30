
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
__key_files = {"name": os.getenv("HOME") + "/PycharmProjects/facial-recognition/keys/embedding_keys.txt",
               "embedding": os.getenv("HOME") + "/PycharmProjects/facial-recognition/keys/name_keys.txt"}
__bit_encryption = 16

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
  open(__key_files[key_type], "w").close()
  with open(__key_files[key_type], "wb") as keys:
    key = get_random_bytes(__bit_encryption)
    keys.write(key)

@require_permission
def generate_cipher(key_type):
  key = get_key(key_type)
  cipher = AES.new(key, AES.MODE_EAX)
  with open(__key_files[key_type], "ab") as keys:
    keys.write(cipher.nonce)
  return cipher

# RETRIEVALS
@require_permission
def get_key(key_type):
  with open(__key_files[key_type], "rb") as keys:
    return b"".join(keys.readlines())[:__bit_encryption]

@require_permission
def get_nonce(key_type, position):
  with open(__key_files[key_type], "rb") as keys:
    joined_nonces = b"".join(keys.readlines())[__bit_encryption:]
    nonce = joined_nonces[position * __bit_encryption:(position + 1) * __bit_encryption]
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
      name = decrypt(bytes([ord(c) for c in encrypted_name]), key_type="name", position=nonce_pos).decode("utf8")

      decrypted[name] = embed

    return decrypted

if __name__ == "__main__":
  data = {"person1": [0.1, 0.2, 0.3], "person2": [0.4, 0.5, 0.7]}
  print(DataEncryption.decrypt_data(DataEncryption.encrypt_data(data)))
  print(data)
  print("it works!")

  # test = "testing!"
  # enc = "".join([chr(c) for c in list(encrypt(bytes(test, "utf8"), cipher))])
  # print(decrypt(bytes([ord(c) for c in enc]), get_key()).decode("utf8"))

  # key = get_key()
  #
  # original = [[0.3, 0.4, 0.5], [0.2, 0.3, 0.4]]
  # encrypted = []
  #
  # for arr in original:
  #   buf = bytes(struct.pack("%sd" % len(arr), *arr))
  #
  #   cipher = generate_cipher()
  #   encrypted.append(list(encrypt(buf, cipher)))
  #
  # for num, arr in enumerate(encrypted):
  #   decrypted = decrypt(bytes(arr), key, num)
  #   fin = list(struct.unpack("%sd" % (len(decrypted) // 8), decrypted))
  #   assert fin == original[num]
