import platform
import sys

if platform.machine() == "arm64":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")

sys.path.insert(1, "../")
from facenet import FaceNet


if __name__ == "__main__":
    facenet = FaceNet()
    facenet.real_time_recognize(detector="trt-mtcnn", graphics=False)
