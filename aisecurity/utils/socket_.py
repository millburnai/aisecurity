import json

from aisecurity.facenet import FaceNet

try:
    import thread
except ImportError:
    import _thread as thread

def on_message(socket, msg):
    return json.loads(msg)

def on_error(socket, error):
    print(error)

def on_close(socket):
    pass

def on_open(socket, **kwargs):
    # FIXME: socket is not a param of real_time_recognize
    facenet = FaceNet()
    thread.start_new_thread(facenet.real_time_recognize(**kwargs, socket=socket))
