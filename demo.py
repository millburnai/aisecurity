
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

from facenet import *

HOME = os.getenv("HOME")

print("Loading facial recognition system...")
facenet = FaceNet(HOME + "/PycharmProjects/facial-recognition/models/facenet_keras.h5")

print("Loading encrypted database...")
facenet.set_data(Preprocessing.retrieve_embeds(HOME + "/PycharmProjects/facial-recognition/images/encrypted.json"))

fig = plt.gcf()
plt.imshow(imread(HOME + "/PycharmProjects/facial-recognition/images/data_example.png"))
plt.title("Example data point from database")
plt.axis("off")
fig.canvas.set_window_title("Facial recognition demo")
plt.show()

input("Press any key to continue:")

facenet.real_time_recognize()