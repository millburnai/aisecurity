import aisecurity.utils.connection as connection
import websocket


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://172.31.217.136:8000/v1/guard/live",
                              on_message = lambda ws,msg: connection.on_message(ws, msg),
                              on_error = connection.on_error,
                              on_close = connection.on_close)
    ws.on_open = connection.on_open
    ws.run_forever()