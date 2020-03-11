"""

"aisecurity.samples.paths"

Common paths used throughout the repository.

"""

import json
import os
import subprocess

import aisecurity

# download stuff and set up stuff
subprocess.call(["make_config.sh"], shell=True)

# paths
CONFIG_HOME = os.path.expanduser("~") + "/.aisecurity"
HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "")

CONFIG = json.load(open(CONFIG_HOME + "/aisecurity.json"))

DATABASE = CONFIG["database_location"]
DATABASE_INFO = json.load(open(CONFIG["database_info"]))

NAME_KEYS = CONFIG["name_keys"]
EMBEDDING_KEYS = CONFIG["embedding_keys"]

DEFAULT_MODEL = CONFIG["default_model"]
