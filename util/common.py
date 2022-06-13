"""Common constants used throughout the repo.
"""

import json
import importlib
import os
import platform
import warnings

REPLACE = "[proj-root]"
CWD = os.path.join(os.path.dirname(__file__), "../")

CONFIG_HOME = os.path.join(CWD, "config")
with open(CONFIG_HOME + "/config.json", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

DB_LOB = CONFIG["database"].replace(REPLACE, CWD)

NAME_KEY_PATH = CONFIG["name_keys"].replace(REPLACE, CWD)
EMBED_KEY_PATH = CONFIG["embedding_keys"].replace(REPLACE, CWD)

DEFAULT_MODEL = CONFIG["default_model"].replace(REPLACE, CWD)
SERVER = CONFIG.get("server", {})
WIFI = {
    "network": SERVER.get("network"),
    "password": SERVER.get("password"),
}
IP = SERVER.get("ip")
IFACE = SERVER.get("iface")

OS = platform.system()
if OS == "Windows":
    warnings.warn(
        "Some features are not supported for Windows OS: "
        "TensorRT and automatic wifi connection"
    )

ON_CUDA = not bool(os.system("command -v nvcc > /dev/null"))
ON_JETSON = platform.machine() == "aarch64"

HAS_RS = importlib.util.find_spec("pyrealsense2") is not None


def name_cleanup(name: str) -> str:
    return name.split("-")[0]
