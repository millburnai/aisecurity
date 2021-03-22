import platform
import sys

if platform.machine() == "arm64":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")

sys.path.insert(1, "../")
from facenet import FaceNet


if __name__ == "__main__":
    on_jetson = platform.machine() == "aarch64"
    try:
        facenet = FaceNet(classifier="svm", fp16=on_jetson)
    except TypeError:
        facenet = FaceNet(classifier="svm")
    facenet.real_time_recognize(detector="trt-mtcnn" if on_jetson else "mtcnn",
                                graphics=not on_jetson)
