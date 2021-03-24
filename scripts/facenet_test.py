import os
import platform
import sys

if platform.machine() == "arm64":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")

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
    facenet.real_time_recognize(detector="trt-mtcnn" if cuda else "mtcnn",
                                graphics=not jetson)
