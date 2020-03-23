"""

"aisecurity.utils.connection"

Handles connection to websocket.

"""

import gc
import json

import websocket


################################ Setup ###############################
SOCKET = None
SOCKET_ADDRESS = None

RECV = None


################################ Websocket ###############################
def init(socket):
    global SOCKET, SOCKET_ADDRESS

    gc.collect()

    try:
        websocket.enableTrace(True)

        SOCKET_ADDRESS = socket

        SOCKET = websocket.create_connection(socket)
        SOCKET.send(json.dumps({"id": "1"}))

        print("[DEBUG] Connected to server")

    except Exception as e:
        print("[DEBUG]", e)
        init(socket)


def send(best_match):
    global SOCKET, RECV

    try:
        SOCKET.send(json.dumps({"best_match": best_match}))
        print("[DEBUG] Sending via websocket...")

        RECV = json.loads(SOCKET.recv())

    except Exception as e:
        print("[DEBUG]", e)
        init(SOCKET_ADDRESS)
        send(best_match)


def reset():
    global RECV

    RECV = None
