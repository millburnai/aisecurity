
"""

"paths.py"

Common paths used throughout the repository.

"""

import os
import aisecurity

HOME = os.getenv("HOME")
if os.path.exists(HOME + "/PycharmProjects/aisecurity"): # for development
  HOME += "/PycharmProjects/aisecurity/aisecurity"
elif os.path.exists(HOME + "/Desktop/aisecurity"):
  HOME += "/Desktop/aisecurity/aisecurity"
else:
  try: # for deployment
    HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "/")
  except AttributeError:
    raise FileNotFoundError("aisecurity repository not found")