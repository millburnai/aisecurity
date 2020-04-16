"""

"aisecurity.samples.demo"

Demonstration of facial recognition system.

"""

import tensorflow as tf
from termcolor import cprint

from aisecurity.facenet import FaceNet
from aisecurity.utils.paths import DEFAULT_MODEL


def demo(path=DEFAULT_MODEL, dist_metric="zero", logging=None, dynamic_log=True,  pbar=False, resize=None, flip=0,
         detector="both", data_mutable=True, socket="ws://67.205.155.37:8000/v1/nano", rotations=None, device=0,
         allow_gpu_growth=False):

    if allow_gpu_growth:
        tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))).__enter__()

    # demo
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])

    facenet = FaceNet(path)

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(
        dist_metric=dist_metric, logging=logging, dynamic_log=dynamic_log, resize=resize, pbar=pbar, flip=flip,
        detector=detector, data_mutable=data_mutable, socket=socket, rotations=rotations, device=device
    )


if __name__ == "__main__":
    import argparse


    # TYPE CASTING
    def to_int(string):
        try:
            return int(string)
        except TypeError:
            raise argparse.ArgumentTypeError("int expected")

    def list_of_ints(string):
        try:
            return [int(val) for val in string.split(",")]
        except TypeError:
            raise argparse.ArgumentTypeError("int list expected")

    def str_or_int(string):
        try:
            return int(string)
        except TypeError:
            return string


    # ARG PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", help="path to facenet model (default: ~/.aisecurity/models/ms_celeb_1m.h5)",
                        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dist_metric", help="distance metric (default: auto)", type=str, default="auto")
    parser.add_argument("--logging", help="logging type, mysql or firebase (default: None)", type=str, default=None)
    parser.add_argument("--dynamic_log", help="use this flag to use dynamic database", action="store_true")
    parser.add_argument("--pbar", help="use this flag to use progress bar", action="store_true")
    parser.add_argument("--flip", help="flip method: +1 = +90º rotation (default: 0)", type=to_int, default=0)
    parser.add_argument("--resize", help="resize frame for faster runtime (default: None)", type=float, default=None)
    parser.add_argument("--detector", help="type of face detector (default: both)", type=str, default="both")
    parser.add_argument("--data_mutable", help="use this flag to allow a mutable db", action="store_true")
    parser.add_argument("--socket", help="websocket address (default: None)", type=str, default=None)
    parser.add_argument("--rotations", help="rotations to be applied to face (-1 is horizontal flip) (default: None)",
                        type=list_of_ints, default=None)
    parser.add_argument("--device", help="video file to read from (default: 0)", type=str_or_int, default=0)
    parser.add_argument("--allow_gpu_growth", help="use this flag to use GPU growth", action="store_true", default=0)
    args = parser.parse_args()


    # DEMO
    demo(
        path=args.path_to_model, dist_metric=args.dist_metric, logging=args.logging, dynamic_log=args.dynamic_log,
        pbar=args.pbar,  flip=args.flip, resize=args.resize, detector=args.detector, data_mutable=args.data_mutable,
        socket=args.socket, rotations=args.rotations, device=args.device, allow_gpu_growth=args.allow_gpu_growth
    )
