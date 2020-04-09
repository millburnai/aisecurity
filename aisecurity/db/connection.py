"""

"aisecurity.db.connection"

Handles connection to websocket.

"""

import functools
import gc
import json

import websocket


################################ Setup and helpers ###############################

# GLOBALS
FAIL_THRESHOLD = 3

SOCKET = None
SOCKET_ADDRESS = None

RECV = None


# DECORATORS
def check_fail(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        failures = 0
        while failures < FAIL_THRESHOLD:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print("[ERROR] {} ({})".format(e, failures))
                _connect(SOCKET_ADDRESS)
                failures += 1

        print("[ERROR] fail threshold passed".format(func))
        return False

    return _func


################################ Websocket ###############################

# CONNECT TO WEBSOCKET
def _connect(socket):
    global SOCKET, SOCKET_ADDRESS

    try:
        gc.collect()
        websocket.enableTrace(True)

        SOCKET_ADDRESS = socket
        SOCKET = websocket.create_connection(socket)
        send(id="1")

        return True

    except Exception:
        return False


# DECORATED FUNCS
@check_fail
def init(socket):
    if not _connect(socket):
        raise ConnectionError("websocket connection failed")
    else:
        print("[DEBUG] Connected to server")


@check_fail
def send(**kwargs):
    global SOCKET, RECV

    SOCKET.send(json.dumps(kwargs))
    print("[DEBUG] Sending via websocket...")


@check_fail
def receive():
    global RECV

    RECV = json.loads(SOCKET.recv())
