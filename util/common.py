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

if "name keys" in CONFIG or "embedding_keys" in CONFIG:
    raise ValueError(
        "Using local key paths is deprecated and unsafe. Upgrade"
        " by using KEYS_ID instead. Try fixing by downloading the "
        "new config directory from 2022-2023."
    )

KEYS_ID = CONFIG["keys_id"]
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
