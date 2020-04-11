"""

"aisecurity.samples.demo"

Demonstration of facial recognition system.

"""

import tensorflow as tf
from termcolor import cprint

from aisecurity.facenet import FaceNet
from aisecurity.utils.paths import DEFAULT_MODEL


def demo(path=DEFAULT_MODEL, dist_metric="zero", logging=None, dynamic_log=True, picam=True, graphics=True,
         pbar=False, resize=None, flip=0, device=0, detector="mtcnn", data_mutable=True,
         socket="ws://67.205.155.37:8000/v1/nano", allow_gpu_growth=False):

    if allow_gpu_growth:
        tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))).__enter__()

    # demo
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])

    facenet = FaceNet(path)

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(
        dist_metric=dist_metric, logging=logging, dynamic_log=dynamic_log, picam=picam, graphics=graphics,
        resize=resize, pbar=pbar, flip=flip, device=device, detector=detector, data_mutable=data_mutable, socket=socket, rotations=rotations
    )


if __name__ == "__main__":
    import argparse


    # TYPE CASTING
    def bounded_float(string):
        if 0. <= float(string) <= 1.:
            return float(string)
        else:
            raise argparse.ArgumentTypeError("float between 0 and 1 expected")

    def to_int(string):
        try:
            return int(string)
        except TypeError:
            raise argparse.ArgumentTypeError("integer expected")


    # ARG PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", help="path to facenet model (default: ~/.aisecurity/models/ms_celeb_1m.h5)",
                        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dist_metric", help="distance metric (default: auto)", type=str, default="auto")
    parser.add_argument("--logging", help="logging type, mysql or firebase (default: None)", type=str, default=None)
    parser.add_argument("--dynamic_log", help="use this flag to use dynamic database", action="store_true")
    parser.add_argument("--no_graphics", help="use this flag to turn off graphics", action="store_true")
    parser.add_argument("--picam", help="use this flag to use a Picamera", action="store_true")
    parser.add_argument("--pbar", help="use this flag to use progress bar", action="store_true")
    parser.add_argument("--flip", help="flip method: +1 = +90º rotation (default: 0)", type=to_int, default=0)
    parser.add_argument("--resize", help="resize frame for faster recognition (default: None)", type=bounded_float,
                        default=None)
    parser.add_argument("--device", help="camera device (default: 0)", type=to_int, default=0)
    parser.add_argument("--detector", help="type of face detector (default: mtcnn)", type=str, default="mtcnn")
    parser.add_argument("--data_mutable", help="use this flag to allow a mutable db", action="store_true")
    parser.add_argument("--allow_gpu_growth", help="use this flag to use GPU growth", action="store_true")
    parser.add_argument("--socket", help="websocket address (default: None)", type=str, default=None)
    args = parser.parse_args()


    # DEMO
    demo(
        path=args.path_to_model, dist_metric=args.dist_metric, logging=args.logging, dynamic_log=args.dynamic_log,
        picam=args.picam, graphics=not args.no_graphics, pbar=args.pbar,  flip=args.flip, resize=args.resize,
        device=args.device, detector=args.detector, data_mutable=args.data_mutable,
        allow_gpu_growth=args.allow_gpu_growth, socket=args.socket
    )
