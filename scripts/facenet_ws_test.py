import sys
sys.path.insert(1, "../")

from facenet import FaceNet
from util.websocket import WebSocket 


if __name__ == "__main__":
    facenet = FaceNet()
    ws = WebSocket("10.56.9.186:8000", 1, facenet)
    ws.connect()
