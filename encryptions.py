
"""

"encryptions.py"

AES encryption for the image database.

"""

import os
import functools

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
      raise OSError("permission denied or file does not exist")
  return _func

# MUTATORS AND RETRIEVERS
@require_permission
def generate_key():
  open(__key_file, "w").close()
  with open(__key_file, "wb") as keys:
    key = get_random_bytes(__bit_encryption)
    keys.write(key)

@require_permission
def get_key():
  with open(__key_file, "rb") as keys:
    return b"".join(keys.readlines())[:__bit_encryption]

@require_permission
def generate_cipher():
  key = get_key()
  cipher = AES.new(key, AES.MODE_EAX)
  with open(__key_file, "wb") as keys:
    keys.write(key)
    keys.write(cipher.nonce)
  return cipher

# ENCRYPT AND DECRYPT
@require_permission
def encrypt(data, cipher):
  cipher_text, tag = cipher.encrypt_and_digest(bytearray(data, encoding="utf8"))
  return cipher_text

def decrypt(cipher_text, key):
  with open(__key_file, "rb") as keys:
    joined = b"".join(keys.readlines())
    nonce = joined[len(joined) - __bit_encryption:]
  decrypt_cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
  plaintext = decrypt_cipher.decrypt(cipher_text)
  try:
    return plaintext.decode("utf8")
  except ValueError:
    raise ValueError("Key incorrect or message corrupted")

if __name__ == "__main__":
  generate_key()
  data = "it works!"
  for i in range(1000):
    cipher = generate_cipher()
    encrypted = encrypt(data, cipher)
    assert decrypt(encrypted, get_key()) == data