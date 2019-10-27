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
def demo():
    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet = FaceNet(CONFIG_HOME + "/models/ms_celeb_1m.h5")

    cprint("\nLoading encrypted database", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet.set_data(Preprocessing.retrieve_embeds(DATABASE, encrypted="names"))

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(use_log=True)


if __name__ == "__main__":
    demo()
