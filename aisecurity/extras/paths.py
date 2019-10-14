"""

"aisecurity.extras.paths"

Common paths used throughout the repository.

"""

import json
import os

import aisecurity


HOME = os.getenv("HOME")
if os.path.exists(HOME + "/PycharmProjects/aisecurity"):  # for development
    HOME += "/PycharmProjects/aisecurity/aisecurity"
elif os.path.exists(HOME + "/Desktop/aisecurity"):
    HOME += "/Desktop/aisecurity/aisecurity"
else:
    try:  # for deployment
        HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "")
    except AttributeError:
        raise FileNotFoundError("aisecurity repository not found")

CONFIG = json.load(open(HOME + "/config.json"))

DATABASE = os.getenv("HOME") + CONFIG["database_location"]

KEY_DIR = os.getenv("HOME") + CONFIG["key_directory"]
KEY_FILE = os.getenv("HOME") + CONFIG["key_location"]
