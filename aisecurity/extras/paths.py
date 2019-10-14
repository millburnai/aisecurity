"""

"aisecurity.extras.paths"

Common paths used throughout the repository.

"""

import json
import os
import subprocess
from termcolor import cprint

import aisecurity


HOME = os.getenv("HOME")
if os.path.exists(HOME + "/PycharmProjects/aisecurity"):  # for development
    CONFIG = json.load(open(HOME + "/PycharmProjects/aisecurity/.aisecurity/aisecurity.json"))
    HOME += "/PycharmProjects/aisecurity/aisecurity"
    subprocess.call([os.getenv("HOME") + "/PycharmProjects/aisecurity/.aisecurity/update_config.sh"])
elif os.path.exists(HOME + "/Desktop/aisecurity"):
    CONFIG = json.load(open(HOME + "/Desktop/aisecurity/.aisecurity/aisecurity.json"))
    HOME += "/Desktop/aisecurity/aisecurity"
    subprocess.call([os.getenv("HOME") + "/PycharmProjects/aisecurity/.aisecurity/update_config.sh"])
    cprint("LIAM GET PYCHARM", color="red", attrs=["bold"])
else:
    try:  # for deployment
        HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "")
        if not os.path.exists(os.getenv("HOME") + "/.aisecurity/aisecurity.json"):
            subprocess.call([HOME + "/create_config.sh"])
        CONFIG = json.load(open(os.getenv("HOME") + "/.aisecurity/aisecurity.json"))
    except AttributeError:
        raise FileNotFoundError("aisecurity repository not found")


DATABASE = CONFIG["database_location"]

KEY_DIR = CONFIG["key_directory"]
KEY_FILE = CONFIG["key_location"]