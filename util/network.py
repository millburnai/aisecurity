import json
from functools import partial
import sys

from socket import error as SocketError
import websocket

sys.path.insert(1, "../")
from util.common import IP


class WebSocket:

    def __init__(self, ip, id, facenet):
        self.ip = ip
        self.id = id
        self.facenet = facenet

    def on_message(self, ws, message):
        message = json.loads(message)
        print("######### SENDING #########")
        print(message)
        print("###########################")

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def connect(self, **facenet_kwargs):
        def on_open(ws, **kwargs):
            ws.send(json.dumps({"id": str(self.id)}))
            self.facenet.real_time_recognize(**kwargs, socket=ws)

        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(f"ws://{self.ip}/v1/nano",
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        ws.on_open = partial(on_open, **facenet_kwargs)
        ws.run_forever()

    @classmethod
    def run(cls, facenet, id=1, **kwargs):
        def _run():
            ws = cls(IP, id, facenet)
            ws.connect(**kwargs)

        i = 0
        while True:
            i += 1
            print(f"------ RESETTING ({i}) ------\n\n\n")
            try:
                _run()
            except SocketError:
                continue
