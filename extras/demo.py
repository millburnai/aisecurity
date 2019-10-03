
"""

"demo.py"

Demonstration of facial recognition system.

"""

# SETUP
import sys
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
  from extras.paths import Paths
except ModuleNotFoundError:
  # for terminal
  import sys
  from paths import Paths
  sys.path.insert(1, Paths.HOME)

# ACTUAL DEMO
from facenet import *

cprint("\nLoading facial recognition system", attrs=["bold"], end="")
cprint("...", attrs=["bold", "blink"])
facenet = FaceNet(Paths.HOME + "/models/facenet_keras.h5")

cprint("\nLoading encrypted database", attrs=["bold"], end="")
cprint("...", attrs=["bold", "blink"])
facenet.set_data(Preprocessing.retrieve_embeds(Paths.HOME + "/database/encrypted.json"))

input("\nPress ENTER to continue:")

facenet.real_time_recognize(use_log=True)