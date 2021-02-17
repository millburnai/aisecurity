import sys

sys.path.insert(1, "../")
from facenet import FaceNet


if __name__ == "__main__":
    facenet = FaceNet()
    facenet.real_time_recognize(detector="trt-mtcnn", flip=False)
