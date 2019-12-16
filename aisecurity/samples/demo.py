"""

"aisecurity.samples.demo"

Demonstration of facial recognition system.

"""


def demo(model="ms_celeb_1m", path=None, logging="firebase", use_dynamic=True, use_picam=False, use_graphics=True,
         use_lcd=False, resize=None, flip=0, allow_gpu_growth=False, verbose=False):

    # imports
    if not verbose:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        import tensorflow as tf

        try:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except AttributeError:
            tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        import os
        import warnings

        import tensorflow as tf

    if allow_gpu_growth:
        tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))).__enter__()

    from termcolor import cprint

    from aisecurity.facenet import FaceNet
    from aisecurity.utils.dataflow import retrieve_embeds
    from aisecurity.utils.paths import DATABASE, CONFIG_HOME


    # demo
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    try:
        facenet = FaceNet(path if path else CONFIG_HOME + "/models/{}.pb".format(model))
    except (OSError, AssertionError):
        facenet = FaceNet(path if path else CONFIG_HOME + "/models/{}.h5".format(model))

    cprint("\nLoading encrypted database", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet.set_data(retrieve_embeds(DATABASE, encrypted="names"))

    input("\nPress ENTER to continue:")
    facenet.real_time_recognize(logging=logging, use_dynamic=use_dynamic, use_picam=use_picam,
                                use_graphics=use_graphics, resize=resize, use_lcd=use_lcd, flip=flip)


if __name__ == "__main__":
    import argparse


    # TYPE CASTING
    def to_bool(string):
        if string.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif string.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("boolean value expected")

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
    parser.add_argument("--model", help="name of facenet model (default: ms_celeb_1m)", type=str, default="ms_celeb_1m")
    parser.add_argument("--path_to_model", help="path to facenet model (default: ~/.aisecurity/models/ms_celeb_1m)",
                        type=str, default=None)
    parser.add_argument("--logging", help="logging type, mysql or firebase (default: None)", type=none_or_str,
                        default=None)
    parser.add_argument("--use_dynamic", help="use dynamic database (default: True)", type=to_bool, default=True)
    parser.add_argument("--use_picam", help="use Picamera (default: False)", type=to_bool, default=False)
    parser.add_argument("--use_graphics", help="display graphics (default: True)", type=to_bool, default=True)
    parser.add_argument("--use_lcd", help="use LCD display (default: False)", type=to_bool, default=False)
    parser.add_argument("--resize", help="resize frame for faster recognition (default: None)", type=bounded_float,
                        default=None)
    parser.add_argument("--flip", help="flip method: +1 = +90º rotation (default: 0)", type=to_int, default=0)
    parser.add_argument("--allow_gpu_growth", help="GPU growth (default: False)", type=to_bool, default=False)
    parser.add_argument("--verbose", help="suppress warnings and TensorFlow output (default: False)", type=to_bool,
                        default=False)
    args = parser.parse_args()


    # DEMO
    demo(model=args.model, path=args.path_to_model, logging=args.logging, use_dynamic=args.use_dynamic,
         use_picam=args.use_picam, use_graphics=args.use_graphics, resize=args.resize, use_lcd=args.use_lcd,
         flip=args.flip, allow_gpu_growth=args.allow_gpu_growth, verbose=args.verbose)
