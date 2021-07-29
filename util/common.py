"""Common constants used throughout the repo.
"""

import json
import importlib
import os
import platform

REPLACE = "[proj-root]"
CWD = os.path.join(os.path.dirname(__file__), "../")

CONFIG_HOME = os.path.join(CWD, "config")
with open(CONFIG_HOME + "/config.json", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

DB_LOB = CONFIG["database"].replace(REPLACE, CWD)

NAME_KEY_PATH = CONFIG["name_keys"].replace(REPLACE, CWD)
EMBED_KEY_PATH = CONFIG["embedding_keys"].replace(REPLACE, CWD)

DEFAULT_MODEL = CONFIG["default_model"].replace(REPLACE, CWD)

ON_CUDA = not bool(os.system("command -v nvcc > /dev/null"))
ON_JETSON = platform.machine() == "aarch64"

HAS_RS = importlib.util.find_spec("pyrealsense2") is not None
