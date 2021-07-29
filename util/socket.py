import socket
import json

try:
    import thread
except ImportError:
    import _thread as thread


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
        print("### error ###")
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def connect(self):
        def on_open(ws):
            ws.send(json.dumps({"id": str(self.id)}))
            self.facenet.real_time_recognize(detector="trt-mtcnn",
                                             graphics=True,
                                             mtcnn_stride=7,
                                             socket=ws)

        socket.enableTrace(True)
        ws = socket.WebSocketApp("ws://{}/v1/nano".format(self.ip),
                                 on_message=self.on_message,
                                 on_error=self.on_error,
                                 on_close=self.on_close)
        ws.on_open = on_open
        ws.run_forever()
