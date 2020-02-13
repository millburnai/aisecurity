import json
import time
import aisecurity
facenet = aisecurity.FaceNet()

try:
    import thread
except ImportError:
    import _thread as thread


def on_message(socket):
	message = json.loads(message)
	return message

def on_error(socket, error):
    print(error)


def on_close(socket):
    pass

def on_open(socket, **kwargs):
	thread.start_new_thread(facenet.real_time_recognize(kwargs, socket=socket))