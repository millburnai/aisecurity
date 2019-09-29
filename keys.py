
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

__key__ = get_random_bytes(16)

def encrypt(data):
  cipher = AES.new(__key__, AES.MODE_EAX)
  nonce = cipher.nonce

  ciphertext, tag = cipher.encrypt_and_digest(bytearray(data, encoding="utf8"))

  return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag):
  cipher = AES.new(__key__, AES.MODE_EAX, nonce=nonce)
  plaintext = cipher.decrypt(ciphertext)
  try:
    cipher.verify(tag)
    return plaintext.decode("utf8")
  except ValueError:
    raise ValueError("Key incorrect or message corrupted")

if __name__ == "__main__":
  print(decrypt(*encrypt("testing!")))