"""

"aisecurity.samples.paths"

Common paths used throughout the repository.

"""

import json
import os
import subprocess

import aisecurity

subprocess.call(["make_config.sh"], shell=True)
# apparently it's bad practice to use shell=True because security reasons
# if anyone wants to remove shell=True and make it still work on Windows please do so

config_home = os.path.expanduser("~") + "/.aisecurity"
home = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "")

config = json.load(open(config_home + "/aisecurity.json", encoding="utf-8"))

db_loc = config["database_location"]
db_info = json.load(open(config["database_info"], encoding="utf-8"))

name_key_path = config["name_keys"]
embed_key_path = config["embedding_keys"]

default_model = config["default_model"]
