"""

"aisecurity.utils.connection"

Connection and socket for real time recognition.

"""

import json

import websocket

from aisecurity.facenet import FaceNet
from aisecurity.utils.events import in_dev


################################ Autoinit ###############################
try:
    import thread
except ImportError:
    import _thread as thread


################################ Functions ###############################

# SOCKET EVENTS
def on_message(ws, msg):
    return json.loads(msg)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("{} closed".format(ws))


def on_open(ws, **kwargs):
    args = ()

    kwargs["socket"] = ws
    kwargs["use_picam"] = True

    thread.start_new_thread(FaceNet().real_time_recognize, args, kwargs=kwargs)


# SOCKET RECOGNITION
@in_dev("real_time_recognize_socket is in production")
def real_time_recognize_socket(socket_url):
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        socket_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    print("Websocket initialized")

    ws.run_forever()


################################ Testing ###############################
if __name__ == "__main__":
    real_time_recognize_socket("ws://172.31.217.136:8000/v1/guard/live")
