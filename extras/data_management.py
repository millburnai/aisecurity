
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

if __name__ == "__main__":
  print("Nothing for now!")