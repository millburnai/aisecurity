"""

"aisecurity.extras.demo"

Demonstration of facial recognition system.

"""

# SETUP
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ERROR HANDLING
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import tensorflow as tf

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)

# PATH SETUP
try:
    # for Pycharm
    from aisecurity.extras.paths import HOME, DATABASE

except ModuleNotFoundError:
    # for terminal
    import sys
    from paths import HOME, DATABASE

    try:
        sys.path.insert(1, os.getenv("HOME") + "/PycharmProjects/aisecurity/")
    except FileNotFoundError:
        sys.path.insert(1, os.getenv("HOME") + "/Desktop/aisecurity/")

from aisecurity.facenet import *


# ACTUAL DEMO
def demo(model="ms_celeb_1m_trt", path=None, use_log=True, use_dynamic=True):
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet = FaceNet(path if path else CONFIG_HOME + "/models/{}.pb".format(model))

    cprint("\nLoading encrypted database", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet.set_data(retrieve_embeds(DATABASE, encrypted="names"))

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(use_log=use_log, use_dynamic=use_dynamic)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of facenet model")
    parser.add_argument("--path_to_model", help="path to facenet model")
    args = parser.parse_args()

    # default values (for Pycharm calls, where args are assigned a None value)
    if args.model is None:
        args.model = "ms_celeb_1m_trt"
    if args.path_to_model is None:
        args.path_to_model = CONFIG_HOME + "/models/{}.pb".format(args.model)

    demo(model=args.model, path=args.path_to_model)
