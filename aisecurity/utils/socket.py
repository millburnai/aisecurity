import websocket
import json
import time
import aisecurity
facenet = aisecurity.FaceNet()

try:
    import thread
except ImportError:
    import _thread as thread


def on_message():
	message = json.loads(message)
	return message

def on_error(ws, error):
    print(error)


def on_close(ws):
    pass

def on_open()
	thread.start_new_thread(facenet.real_time_recognize())