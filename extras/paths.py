
"""

"paths.py"

Common paths used throughout the repository.

"""

import os

HOME = os.getenv("HOME")
if os.path.exists(HOME + "/PycharmProjects/facial-recognition"):
  HOME += "/PycharmProjects/facial-recognition"
elif os.path.exists(HOME + "/Desktop/facial-recognition"):
  HOME += "/Desktop/facial-recognition"
else:
  raise FileNotFoundError("facial-recognition repository not found. Please move to ~/PycharmProjects or ~/Desktop")