"""

"aisecurity.samples.demo"

Demonstration of facial recognition system.

"""

import tensorflow as tf
from termcolor import cprint

from aisecurity.facenet import FaceNet
from aisecurity.utils.paths import DEFAULT_MODEL


def demo(path=DEFAULT_MODEL, dist_metric="auto", logging=None, use_dynamic=True, use_picam=False, use_graphics=True,
         use_lcd=False, use_keypad=False, resize=None, flip=0, device=0, face_detector="mtcnn", update_static=False,
         allow_gpu_growth=False):

    if allow_gpu_growth:
        tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))).__enter__()

    # demo
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])

    facenet = FaceNet(path)

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(
        dist_metric=dist_metric, logging=logging, use_dynamic=use_dynamic, use_picam=use_picam,
        use_graphics=use_graphics, resize=resize, use_lcd=use_lcd, use_keypad=use_keypad, flip=flip, device=device,
        face_detector=face_detector, update_static=update_static
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

    def none_or_str(string):
        if string.lower() == "none":
            return None
        else:
            return string


    # ARG PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", help="path to facenet model (default: ~/.aisecurity/models/ms_celeb_1m.h5)",
                        type=str, default=None)
    parser.add_argument("--dist_metric", help="distance metric (default: auto)", type=str, default="auto")
    parser.add_argument("--logging", help="logging type, mysql or firebase (default: None)", type=none_or_str,
                        default=None)
    parser.add_argument("--use_dynamic", help="use this flag to use dynamic database", action="store_true")
    parser.add_argument("--no_graphics", help="use this flag to turn off graphics", action="store_true")
    parser.add_argument("--use_picam", help="use this flag to use a Picamera", action="store_true")
    parser.add_argument("--use_lcd", help="use this flag to use LCD display", action="store_true")
    parser.add_argument("--use_keypad", help="use this flag to use keypad", action="store_true")
    parser.add_argument("--flip", help="flip method: +1 = +90º rotation (default: 0)", type=to_int, default=0)
    parser.add_argument("--resize", help="resize frame for faster recognition (default: None)", type=bounded_float,
                        default=None)
    parser.add_argument("--device", help="camera device (default: 0)", type=to_int, default=0)
    parser.add_argument("--face_detector", help="type of face detector (default: mtcnn)", type=str, default="mtcnn")
    parser.add_argument("--update_static", help="use this flag to update static database", action="store_true")
    parser.add_argument("--allow_gpu_growth", help="use this flag to use GPU growth", action="store_true")
    args = parser.parse_args()


    # DEMO
    demo(
        path=args.path_to_model, dist_metric=args.dist_metric, logging=args.logging,
        use_dynamic=args.use_dynamic, use_picam=args.use_picam, use_graphics=not args.no_graphics,
        use_lcd=args.use_lcd, use_keypad=args.use_keypad, flip=args.flip, resize=args.resize, device=args.device,
        face_detector=args.face_detector, update_static=args.update_static, allow_gpu_growth=args.allow_gpu_growth
    )
