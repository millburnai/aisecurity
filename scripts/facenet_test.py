import os
import platform
import sys

sys.path.insert(1, "../")
from facenet import FaceNet


if __name__ == "__main__":
    jetson = platform.machine() == "aarch64"

    print("[DEBUG] checking for CUDA...")
    cuda = not bool(os.system("nvcc --version"))
    if cuda:
        print("[DEBUG] CUDA found... using tensorrt")
    else:
        print("[DEBUG] CUDA not found... defaulting to tensorflow")

    facenet = FaceNet()

    detector = "trt-mtcnn" if cuda else "mtcnn"
    graphics = not jetson
    stride = 7 if jetson else 1

    facenet.real_time_recognize(detector=detector,
                                graphics=graphics,
                                stride=stride)
