"""

"aisecurity.utils.connection"

Handles connection to websocket.

"""

import gc
import json

import websocket

from aisecurity.utils.decorators import check_fail


################################ Setup and helpers###############################
FAIL_THRESHOLD = 3

SOCKET = None
SOCKET_ADDRESS = None

RECV = None


################################ Websocket ###############################
@check_fail(FAIL_THRESHOLD)
def init(socket):
    global SOCKET, SOCKET_ADDRESS

    gc.collect()

    try:
        websocket.enableTrace(True)

        SOCKET_ADDRESS = socket

        SOCKET = websocket.create_connection(socket)
        SOCKET.send(json.dumps({"id": "1"}))

        print("[DEBUG] Connected to server")

        return True

    except Exception as e:
        print("[DEBUG]", e)
        init(socket)

        return False


@check_fail(FAIL_THRESHOLD)
def send(obj):
    global SOCKET, RECV

    try:
        SOCKET.send(json.dumps(obj))
        print("[DEBUG] Sending via websocket...")

        RECV = json.loads(SOCKET.recv())

        return True

    except Exception as e:
        print("[DEBUG]", e)
        init(SOCKET_ADDRESS)
        send(obj)

        return False


def reset():
    global RECV

    RECV = None
