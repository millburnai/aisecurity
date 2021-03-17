import websocket
import json
from facenet import FaceNet
try:
    import thread
except ImportError:
    import _thread as thread
import time

class WebSocket():
    def __init__ (self, ip, id, facenet):
        self.ip = ip
        self.id = id
        self.facenet = facenet

    def on_message(self, ws, message):
        message = json.loads(message)
        print(message)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def connect(self):
        websocket.enableTrace(True)
        ws = websocket.WebSocketApp("ws://{}/v1/nano".format(self.ip),
                              on_message = lambda ws,msg: self.on_message(ws, msg),
                              on_error = self.on_error,
                              on_close = self.on_close)
        def on_open(ws):
            def run(*args):
                ws.send(json.dumps({"id":self.id}))
                self.facenet.real_time_recognize(detector="mtcnn", graphics=True, socket=ws)
            thread.start_new_thread(run, ())
            #print("thread terminating...")
        ws.on_open = on_open
        ws.run_forever()
