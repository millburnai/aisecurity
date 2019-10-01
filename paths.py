
"""

"paths.py"

Common paths used throughout the repository.

"""

import os

class Paths(object):
  HOME = os.getenv("HOME")
  if os.path.exists(HOME + "/PycharmProjects/facial-recognition"):
    HOME += "/PycharmProjects/facial-recognition"
  elif os.path.exists(HOME + "/Desktop/facial-recognition"):
    HOME += "/Desktop/facial-recognition"
  else:
    raise FileNotFoundError("facial-recognition repository not found")

  img_dir = HOME + "/images/database/"

  people = None
  if os.path.exists(img_dir):
    people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
