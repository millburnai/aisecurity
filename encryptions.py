
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
__key_file = os.getenv("HOME") + "/PycharmProjects/facial-recognition/keys.txt"
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
def generate_key():
  open(__key_file, "w").close()
  with open(__key_file, "wb") as keys:
    key = get_random_bytes(__bit_encryption)
    keys.write(key)

@require_permission
def generate_cipher():
  key = get_key()
  cipher = AES.new(key, AES.MODE_EAX)
  with open(__key_file, "ab") as keys:
    keys.write(cipher.nonce)
  return cipher

# RETRIEVALS
@require_permission
def get_key():
  with open(__key_file, "rb") as keys:
    return b"".join(keys.readlines())[:__bit_encryption]

@require_permission
def get_nonce(position):
  with open(__key_file, "rb") as keys:
    joined_nonces = b"".join(keys.readlines())[__bit_encryption:]
    if position == -1:
      position = len(joined_nonces) - 1
    nonce = joined_nonces[position * __bit_encryption:(position + 1) * __bit_encryption]
  return nonce

# ENCRYPT AND DECRYPT
def encrypt(data, cipher):
  cipher_text, tag = cipher.encrypt_and_digest(data)
  return cipher_text

def decrypt(cipher_text, key, position=-1):
  decrypt_cipher = AES.new(key, AES.MODE_EAX, nonce=get_nonce(position))
  return decrypt_cipher.decrypt(cipher_text)

if __name__ == "__main__":

  generate_key()
  key = get_key()

  test = "testing!"
  # enc = "".join([chr(c) for c in list(encrypt(bytes(test, "utf8"), cipher))])
  # print(decrypt(bytes([ord(c) for c in enc]), get_key()).decode("utf8"))

  original = np.random.random((5, 128))
  encrypted = []
  for arr in original:
    arr = arr.tolist()

    buf = bytes(struct.pack("%sd" % len(arr), *arr))
    # test = list(struct.unpack("%sd" % (len(buf) // 8), buf))

    cipher = generate_cipher()
    encrypted.append(list(encrypt(buf, cipher)))

  for num, arr in enumerate(encrypted):
    decrypted = decrypt(bytes(arr), key, num)
    fin = list(struct.unpack("%sd" % (len(decrypted) // 8), decrypted))
    print(fin == original[num])