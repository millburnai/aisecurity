
"""

"data_management.py"

Internal use only-- manages data.

"""

from aisecurity.facenet import *

def redump(json_file, ignore=None):
  data = Preprocessing.retrieve_embeds(HOME + "/database/_decrypted.json")
  with open(json_file, "w") as dump_file:
    json.dump(DataEncryption.encrypt_data(data, ignore=ignore), dump_file, indent=4)
  data = Preprocessing.retrieve_embeds(json_file)
  print(list(data.keys()))

def dump_from_encrypted():
  data = Preprocessing.retrieve_embeds(HOME + "/database/_encrypted.json")

  with open(HOME + "/database/_decrypted.json", "w+") as json_file:
    data = dict((key, val.tolist()) for key, val in data.items())
    json.dump(data, json_file, indent=4)

  data = Preprocessing.retrieve_embeds(HOME + "/database/_decrypted.json")
  print(data.keys())

def dump_new(fp=HOME + "/database/_encrypted.json"):
  facenet = FaceNet(HOME + "/models/facenet_keras.h5")
  Preprocessing.dump_embeds(facenet, HOME + "/database/images/", fp, HOME + "/database/_encrypted.json",
                            ignore_encrypt="embeddings")
  print(Preprocessing.retrieve_embeds(fp).keys())

if __name__ == "__main__":
  # redump(HOME + "/database/_encrypted.json", ignore=["embeddings"])
  data = Preprocessing.retrieve_embeds(HOME + "/database/_encrypted.json")
  print(data.keys())
  print("Nothing for now!")