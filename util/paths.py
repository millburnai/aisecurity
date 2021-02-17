"""Common paths used throughout the package.
"""

import json
import os

CONFIG_HOME = os.path.join(os.path.dirname(__file__), "../config")
with open(CONFIG_HOME + "/config.json", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

DB_LOB = CONFIG["database"]

NAME_KEY_PATH = CONFIG["name_keys"]
EMBED_KEY_PATH = CONFIG["embedding_keys"]

DEFAULT_MODEL = CONFIG["default_model"]
