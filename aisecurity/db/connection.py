"""

"aisecurity.db.connection"

Handles connection to websocket.

"""

import gc
import json

import websocket


################################ Websocket ###############################
class Websocket:

    def __init__(self, socket):
        self.recv = None
        if not self._connect(socket):
            raise ConnectionError("websocket connection failed")
        else:
            print("[DEBUG] connected to server")

    def _connect(self, socket):
        try:
            gc.collect()
            websocket.enableTrace(True)

            self.socket_address = socket
            self.socket = websocket.create_connection(socket)
            self.send(id="1")

            return True

        except Exception:
            return False

    def send(self, **kwargs):
        try:
            self.socket.send(json.dumps(kwargs))
            print("[DEBUG] sending via websocket")
            return True

        except Exception:
            print("[ERROR] send failed")
            return False

    def receive(self):
        self.recv = json.loads(self.socket.recv())
