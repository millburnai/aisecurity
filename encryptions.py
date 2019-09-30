
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
__key_file = "/Users/ryan/PycharmProjects/facial-recognition/keys.txt"
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
def generate_cipher(new_nonce=True):
  key = get_key()
  nonce = get_nonce()
  cipher = AES.new(key, AES.MODE_EAX)
  with open(__key_file, "wb") as keys:
    keys.write(key)
    if new_nonce:
      keys.write(cipher.nonce)
    else:
      keys.write(nonce)
  return cipher

# RETRIEVALS
@require_permission
def get_key():
  with open(__key_file, "rb") as keys:
    return b"".join(keys.readlines())[:__bit_encryption]

@require_permission
def get_nonce():
  with open(__key_file, "rb") as keys:
    joined = b"".join(keys.readlines())
    nonce = joined[len(joined) - __bit_encryption:]
  return nonce

# ENCRYPT AND DECRYPT
def encrypt(data, cipher):
  cipher_text, tag = cipher.encrypt_and_digest(data)
  return cipher_text

def decrypt(cipher_text, key):
  decrypt_cipher = AES.new(key, AES.MODE_EAX, nonce=get_nonce())
  return decrypt_cipher.decrypt(cipher_text)

if __name__ == "__main__":

  generate_key()
  key = get_key()

  test = "testing!"
  cipher = generate_cipher()
  # enc = "".join([chr(c) for c in list(encrypt(bytes(test, "utf8"), cipher))])
  # print(decrypt(bytes([ord(c) for c in enc]), get_key()).decode("utf8"))

  original = list(np.random.random((5, 128)))
  encrypted = []
  for arr in original:
    arr = list(arr)

    buf = struct.pack("%sf" % len(arr), *arr)

    cipher = generate_cipher()
    encrypted.append(encrypt(buf, cipher))
    fin = decrypt(encrypted[-1], key)
    print(np.array_equal(list(struct.unpack("%sf" % (len(fin) // 4), fin)), arr))

  for num, arr in enumerate(encrypted):
    fin = decrypt(arr, get_key())
    print(np.array_equal(list(struct.unpack("%sf" % (len(fin) // 4), fin)), original[num]))