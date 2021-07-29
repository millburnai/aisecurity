import websocket
import json
from functools import partial


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
