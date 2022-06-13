import sys
import argparse

sys.path.insert(1, "../")

from facenet import FaceNet
from util.common import ON_CUDA, ON_JETSON

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="an integer for the accumulator")
    parser.add_argument("--detector", type=str, help="the detector mode")
    parser.add_argument(
        "--graphics", type=bool, help="sets the graphics mode on or off"
    )
    parser.add_argument("--resize", type=str, help="sets the resize to auto by default")
    parser.add_argument(
        "--mtcnn_stride", type=str, help="sets the mtcnn_stride to auto by default"
    )

    args = parser.parse_args()

    if args.detector is None:
        args.detector = "mtcnn"

    if args.graphics is None:
        args.graphics = True

    if args.mtcnn_stride is None:
        args.mtcnn_stride = "auto"

    if args.resize is None:
        args.resize = "auto"

    if args.mode is None:
        args.mode = "cosine"

    if args.mtcnn_stride:
        mtcnn_stride = 7 if ON_JETSON else 3
    else:
        mtcnn_stride = 1

    if args.resize:
        resize = 1 if ON_JETSON else 0.6
    else:
        resize = 1

    facenet = FaceNet()
    facenet.real_time_recognize(
        detector=args.detector,
        graphics=args.graphics,
        mtcnn_stride=mtcnn_stride,
        resize=resize,
        mode=args.mode,
    )
