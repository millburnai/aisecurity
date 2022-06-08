import sys

sys.path.insert(1, "../")

from facenet import FaceNet
from util.common import ON_CUDA, ON_JETSON


if __name__ == "__main__":
    detector = "mtcnn"  # "trt-mtcnn" if ON_CUDA else "mtcnn"
    graphics = True  # not ON_JETSON
    mtcnn_stride = 7 if ON_JETSON else 3
    resize = 1# if ON_JETSON else 0.6

    facenet = FaceNet()
    facenet.real_time_recognize(
        width=846, height=1156,
        detector=detector, graphics=graphics, mtcnn_stride=mtcnn_stride, resize=resize
    )
