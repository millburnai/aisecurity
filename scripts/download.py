import json
import os
import platform
import sys
import re
import dropbox

from pathlib import Path
from tqdm import tqdm

def get_files(dbx, prefix, ignore) -> None:
    has_more = True
    cursor = None

    while has_more:
        if cursor:
            res = dbx.files_list_folder_continue(cursor)
        else:
            res = dbx.files_list_folder(prefix)

        for elem in res.entries:
            if not ignore.match(elem.path_display):
                if isinstance(elem, dropbox.files.FolderMetadata):
                    yield from get_files(dbx, elem.path_display, ignore)
                else:
                    yield elem.path_display

        has_more = res.has_more
        cursor = res.cursor

if __name__ == "__main__":
    dest = "../config"
    dbx_prefix = "/aisecurity/2020-2021/config"
    ignore = re.compile(r".*photos$|^.*\.(jpg|png)$")

    try:
        token = sys.argv[1]
    except IndexError:
        print("USAGE: python download.py [token]")
        sys.exit(1)

    dbx = dropbox.Dropbox(token)
    display_name = dbx.users_get_current_account().name.display_name
    print(f"[DEBUG] logged into account name: \"{display_name}\"")

    urls = list(get_files(dbx, dbx_prefix, ignore))
    for url in tqdm(urls):
        path = url.replace(dbx_prefix, dest)
        head, tail = os.path.split(path)

        Path(head).mkdir(parents=True, exist_ok=True)
        dbx.files_download_to_file(path, url)

    print(f"[DEBUG] downloaded to {dest}")

    with open("../config/config.json", "r", encoding="utf8") as f:
        config = json.load(f)
        prefix, ext = os.path.splitext(config["default_model"])

    on_gpu = not bool(os.system("command -v nvcc > /dev/null"))
    on_jetson = platform.machine() == "aarch64"

    if on_jetson or on_gpu:
        print("[DEBUG] changing default backend to tensorrt")
        config["default_model"] = prefix + ".engine"
        with open("../config/config.json", "w", encoding="utf8") as f:
            json.dump(config, f)

        for i in range(1, 4):
            os.rename(f"../config/models/det{i}.engine",
                      f"../util/trt_mtcnn_plugin/mtcnn/det{i}.engine")

    else:
        backends = {".pb": "tensorflow", ".tflite": "tflite"}
        print(f"[DEBUG] default backend is {backends[ext]}")
