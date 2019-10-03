
"""

"data_management.py"

Internal use only-- manages data.

"""

from facenet import *

def redump(json_file):
  data = Preprocessing.retrieve_embeds(Paths.HOME + "/database/_processed.json", False)
  with open(json_file, "w") as dump_file:
    json.dump(DataEncryption.encrypt_data(data), dump_file, indent=4)
  data = Preprocessing.retrieve_embeds(json_file)
  print(list(data.keys()))

def dump_from_encrypted():
  facenet = FaceNet(Paths.HOME + "/models/facenet_keras.h5")
  facenet.set_data(Preprocessing.retrieve_embeds(Paths.HOME + "/database/encrypted.json"))

  with open(Paths.HOME + "/database/_processed.json", "w+") as json_file:
    data = dict((key, val.tolist()) for key, val in facenet.data.items())
    json.dump(data, json_file, indent=4)

  facenet.set_data(Preprocessing.retrieve_embeds(Paths.HOME + "/database/_processed.json", encrypted=False))
  print(facenet.data.keys())

def dump_new(fp=Paths.HOME + "/database/encrypted.json"):
  facenet = FaceNet(Paths.HOME + "/models/facenet_keras.h5")
  Preprocessing.dump_embeds(facenet, Paths.HOME + "/database/images/", fp, Paths.HOME + "/database/encrypted.json")
  print(Preprocessing.retrieve_embeds(fp).keys())

if __name__ == "__main__":
  dump_new()
  print("Nothing for now!")