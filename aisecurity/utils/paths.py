"""

"aisecurity.samples.paths"

Common paths used throughout the repository.

"""

import json
import os
import subprocess

import aisecurity

CONFIG_HOME = os.getenv("HOME") + "/.aisecurity"
HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "")
if not os.path.exists(CONFIG_HOME + "/aisecurity.json"):
    subprocess.call(["make_config.sh"])
CONFIG = json.load(open(CONFIG_HOME + "/aisecurity.json"))

DATABASE = CONFIG["database_location"]
DATABASE_INFO = json.load(open(CONFIG["database_info"]))

KEY_DIR = CONFIG["key_directory"]
KEY_FILE = CONFIG["key_location"]

NAME_KEYS = json.load(open(KEY_FILE))["names"]
EMBEDDING_KEYS = json.load(open(KEY_FILE))["embeddings"]
