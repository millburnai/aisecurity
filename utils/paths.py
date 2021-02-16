"""Common paths used throughout the package.
"""

import json

config_home = "config"
with open(config_home + "/config.json", encoding="utf-8") as config_file:
    config = json.load(config_file)

db_loc = config["database"]

name_key_path = config["name_keys"]
embed_key_path = config["embedding_keys"]

default_model = config["default_model"]
