"""Main production script for kiosk/aisecurity."""

import sys
sys.path.insert(1, "../")

from facenet import FaceNet
from util.network import WebSocket
from util.common import ON_CUDA, ON_JETSON


if __name__ == "__main__":
    WebSocket.run(FaceNet(),
                  detector="trt-mtcnn" if ON_CUDA else "mtcnn",
                  graphics=not ON_JETSON,
                  mtcnn_stride=7 if ON_JETSON else 3,
                  resize=1 if ON_JETSON else 0.6)
