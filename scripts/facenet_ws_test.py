import platform
import sys

if platform.machine() == "arm64":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")

sys.path.insert(1, "../")
from facenet import FaceNet
from util.websocket import WebSocket 


if __name__ == "__main__":
    facenet = FaceNet()
    ws = WebSocket("127.0.0.1:8000", 1, facenet)
    ws.connect()
