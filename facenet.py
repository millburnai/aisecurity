
"""

"facenet.py"

Facial recognition with FaceNet in Keras.

Paper: https://arxiv.org/pdf/1503.03832.pdf

"""

import os
from time import time
import functools

import matplotlib.pyplot as plt
import keras
from keras import backend as K
import numpy as np
from scipy.spatial import distance
from skimage.transform import resize
from imageio import imread
from mtcnn.mtcnn import MTCNN

# DECORATORS
def timer(message = "Time elapsed"):

  def _timer(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
      start = time()
      result = func(*args, **kwargs)
      print("{}: {}s".format(message, round(time() - start, 3)))
      return result
    return _func

  return _timer

# FACENET
class FaceNet(object):

  # HYPERPARAMETERS
  ALPHA = 1.0

  # INITS
  @timer(message="Model load time")
  def __init__(self, filepath):
    self.k_model = keras.models.load_model(filepath)
    self.data = None # must be filled in by user

  def set_data(self, data):
    self.data = data

  # GENERIC L2 DISTANCE
  def l2_dist(self, img_a, img_b):
    return distance.euclidean(self.data[img_a]["embedding"], self.data[img_b]["embedding"])

  # COMPARE TWO SPECIFIC IMAGES
  def compare(self, img_a, img_b, verbose = True):
    assert self.data, "data must be provided"

    dist = self.l2_dist(img_a, img_b)
    is_same = dist <= FaceNet.ALPHA

    if verbose:
      print("L2 normalized distance: {} -> {} and {} are the same person: {}".format(dist, img_a, img_b, is_same))

      plt.subplot(1, 2, 1)
      plt.imshow(imread(self.data[img_a]["img"]))
      plt.axis("off")

      plt.subplot(1, 2, 2)
      plt.imshow(imread(self.data[img_b]["img"]))
      plt.axis("off")

      plt.suptitle("Same person: {}\n L2 distance: {}".format(is_same, dist))
      plt.show()

    return int(is_same), dist

# IMAGE PREPROCESSING
class Preprocessing(object):

  # HYPERPARAMETERS
  IMG_SIZE = 160

  @staticmethod
  def prewhiten(x):
    if x.ndim == 4:
      axis = (1, 2, 3)
      size = x[0].size
    elif x.ndim == 3:
      axis = (0, 1, 2)
      size = x.size
    else:
      raise ValueError("x must have either 3 or 4 dimensions")

    std_adj = np.maximum(np.std(x, axis=axis, keepdims=True), 1.0 / np.sqrt(size))
    whitened = (x - np.mean(x, axis=axis, keepdims=True)) / std_adj
    return whitened

  @staticmethod
  def align_imgs(filepaths, margin):
    detector = MTCNN()

    def align_img(path):
      img = imread(path)
      faces = detector.detect_faces(img)[0]["box"]
      assert len(faces) != 0, "face was not found in {}".format(path)

      x, y, width, height = faces
      cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
      resized = resize(cropped, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE), mode="reflect")
      return resized
  
    return np.array([align_img(path) for path in filepaths])

  @staticmethod
  def embed(facenet, filepaths, margin = 10, batch_size = 1):
    aligned_imgs = Preprocessing.prewhiten(Preprocessing.align_imgs(filepaths, margin))
    raw_embeddings = facenet.k_model.predict(aligned_imgs, batch_size=batch_size)

    l2_normalize = lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), K.epsilon()))
    normalized_embeddings = l2_normalize(raw_embeddings)

    return normalized_embeddings

  @staticmethod
  @timer(message="Data preprocessing time")
  def load(facenet, img_dir, people):
    data = {}
    for person in people:
      person_dir = img_dir + person
      img_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if not f.endswith(".DS_Store")]
      embeddings = Preprocessing.embed(facenet, img_paths)
      for index, path in enumerate(img_paths):
        data["{}{}".format(person, index)] = {"img": path, "embedding": embeddings[index]}
    return data

if __name__ == "__main__":
  # PATHS
  HOME = os.getenv("HOME")
  img_dir = HOME + "/PycharmProjects/ai-security/images/"
  people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store")]

  # NETWORK INIT
  facenet = FaceNet(HOME + "/PycharmProjects/ai-security/models/facenet_keras.h5")
  facenet.set_data(Preprocessing.load(facenet, img_dir, people))

  # TESTING
  start = time()

  my_imgs = []
  for person in people:
    for index in range(len([f for f in os.listdir(img_dir + person) if not f.endswith(".DS_Store")])):
      my_imgs.append("{}{}".format(person, index))

  count = 0
  for img_a in my_imgs:
    for img_b in my_imgs:
      if not np.array_equal(img_a, img_b):
        facenet.compare(img_a, img_b)
        count += 1

  print("Average time per comparison: {}s".format(round((time() - start) / count, 3)))