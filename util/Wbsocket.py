import websocket
import json
import threading
import sys
import time
sys.path.insert(1, "../")
from facenet import FaceNet
class Wbsocket:
    def on_message(self, ws, message):
        message = json.loads(message)
        print(message)
    def on_error(self, ws, error):
        print(error)
    def on_close(self, ws):
        print("### closed ###")
    def on_open(self, ws):
        def run(*args):
            ws.send(json.dumps({"idd":"1"}))
            time.sleep(1)
        #print("thread terminating...")
        facenet = FaceNet(classifier="svm")
        Thread = threading.Thread(target=facenet.real_time_recognize, kwargs=dict(detector="mtcnn",graphics=True))
        Thread.start()
    def __init__(self, ip):
        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(ip,
                                on_message = lambda ws,msg: self.on_message(ws, msg),
                                on_error = self.on_error,
                                on_close = self.on_close)
        ws.on_open = self.on_open
        ws.run_forever()
