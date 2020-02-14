import aisecurity.utils.socket_ as socket_
import websocket

from aisecurity.utils.events import in_dev

@in_dev("real_time_recognize_socket is in production")
def real_time_recognize_socket(socket_url)
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket_url,
                              on_message = lambda ws,msg: socket_.on_message(ws, msg),
                              on_error = socket_.on_error,
                              on_close = socket_.on_close)
    ws.on_open = socket_.on_open(ws)
    ws.run_forever()
