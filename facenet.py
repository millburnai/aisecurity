
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
import cv2
from skimage.transform import resize
from imageio import imread
from mtcnn.mtcnn import MTCNN

# ERROR HANDLING
def suppress_tf_warnings():
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  import warnings
  warnings.simplefilter(action="ignore", category=FutureWarning)
  warnings.simplefilter(action="ignore", category=UserWarning)

  import tensorflow as tf
  try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)

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
    self.k_nn = None # instantiated in functions

  # MUTATORS AND RETRIEVERS
  def set_data(self, data):
    self.data = data

  def _set_knn(self):
    # TODO: speed up recognize by replacing complete search of dataset with k-NN classifier
    pass
    # normalize_embeds = lambda embed_dict: np.array([embed_dict[person]["embedding"] for person in embed_dict])
    #
    # k_nn = keras.layers.Dense(name="k-nn", units=self.k_model.output_size, use_bias=False)
    # k_nn.set_weights(normalize_embeds(self.data))
    #
    # self.k_nn = keras.Model(input=[k_nn.input], output=k_nn.output)

  def get_facenet(self):
    return self.k_model

  # GENERIC L2 DISTANCE
  def l2_dist(self, img_a, img_b):
    try:
      return np.linalg.norm(self.data[img_a]["embedding"] - self.data[img_b]["embedding"])
    except TypeError:
      return np.linalg.norm(img_a - self.data[img_b]["embedding"]) # assumes img_a is a precomputed embedding

  def predict(self, filepaths, batch_size=1, faces=None):
    return Preprocessing.embed(self.k_model, filepaths, batch_size = batch_size, faces=faces)

  # COMPARE TWO SPECIFIC IMAGES
  def compare(self, img_a, img_b, verbose=True):
    assert self.data, "data must be provided"

    dist = self.l2_dist(img_a, img_b)
    is_same = dist <= FaceNet.ALPHA

    if verbose:
      print("L2 distance: {} -> {} and {} are the same person: {}".format(dist, img_a, img_b, is_same))
      self.disp_imgs(img_a, img_b, title = "Same person: {}\n L2 distance: {}".format(is_same, dist))

    return int(is_same), dist

  # FACIAL RECOGNITION HELPER
  @timer(message="Recognition time")
  def _recognize(self, img, verbose=True, faces=None):
    assert self.data, "data must be provided"

    def find_min_key(dict_):
      minimum = (None, np.float("inf"))
      for key, val in dict_.items():
        if val < minimum[1]:
          minimum = (key, val)
      return minimum[0]

    embedding = self.predict([img], faces=faces)
    avgs = {}
    for person in self.data:
      if person[:-1] not in avgs:
        avgs[person[:-1]] = self.l2_dist(embedding, person)
    best_match = find_min_key(avgs)

    if verbose:
      if avgs[best_match] <= FaceNet.ALPHA:
        print("Your image is a picture of \"{}\": L2 distance of {}".format(best_match, avgs[best_match]))
      else:
        print("Your image is not in the database. The best match is \"{}\" with an L2 distance of ".format(
          best_match, avgs[best_match]))
      self.disp_imgs(img, "{}0".format(best_match), title="Best match: {}\nL2 distance: {}".format(
        best_match, avgs[best_match]))

    return int(avgs[best_match] <= FaceNet.ALPHA), best_match, avgs[best_match]

  # FACIAL RECOGNITION
  def recognize(self, img):
    return self._recognize(img, verbose=True, faces=None)

  # REAL TIME RECOGNITION (DEMO)
  def realtime_recognize(self):
    if self.k_nn is None:
      self._set_knn()

    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    while True:
      _, frame = cap.read()
      result = detector.detect_faces(frame)

      if result:
        for person in result:
          faces = person["box"]
          x, y, height, width = faces

          is_recognized, best_match, l2_dist = self._recognize(frame, verbose=False, faces=faces)
          color = (0, 255, 0) if is_recognized else (0, 0, 255) # green if is_recognize else red

          cv2.rectangle(frame, (x, y), (x + height, y + width), color, thickness=2)
          cv2.putText(frame, best_match, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=color)

      cv2.imshow("frame", frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    cv2.destroyAllWindows()

  # DISPLAYING
  def disp_imgs(self, img_a, img_b, title = None):
    plt.subplot(1, 2, 1)
    try:
      plt.imshow(imread(self.data[img_a]["path"]))
    except KeyError:
      plt.imshow(imread(img_a))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    try:
      plt.imshow(imread(self.data[img_b]["path"]))
    except KeyError:
      plt.imshow(imread(img_b))
    plt.axis("off")

    if title is not None:
      plt.suptitle(title)

    plt.show()

# IMAGE PREPROCESSING
class Preprocessing(object):

  # HYPERPARAMETERS
  IMG_SIZE = 160

  @staticmethod
  def whiten(x):
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
  def align_imgs(filepaths, margin, faces=None):
    if not faces:
      detector = MTCNN()

    def align_img(path, faces=None):
      try:
        img = imread(path)
      except OSError:
        img = path # TODO: restructure so that you don't have to catch OSError-- let Preprocessing accept numpy arrays
                   #       as images

      if not faces:
        found = detector.detect_faces(img)
        assert len(found) != 0, "face was not found in {}".format(path)
        faces = found[0]["box"]

      x, y, width, height = faces
      cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
      resized = resize(cropped, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE), mode="reflect")
      return resized

    return np.array([align_img(path, faces=faces) for path in filepaths])

  @staticmethod
  def embed(facenet, filepaths, margin=15, batch_size=1, faces=None):
    aligned_imgs = Preprocessing.whiten(Preprocessing.align_imgs(filepaths, margin, faces=faces))
    raw_embeddings = facenet.predict(aligned_imgs, batch_size=batch_size)

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
        data["{}{}".format(person, index)] = {"path": path, "embedding": embeddings[index]}
    return data

if __name__ == "__main__":
  suppress_tf_warnings()

  # PATHS
  HOME = os.getenv("HOME")

  img_dir = HOME + "/PycharmProjects/facial-recognition/images/database/"
  people = ["ryan", "liam"] # [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store")]

  # NETWORK INIT
  facenet = FaceNet(HOME + "/PycharmProjects/facial-recognition/models/facenet_keras.h5")
  facenet.set_data(Preprocessing.load(facenet.get_facenet(), img_dir, people))

  # UNIT TESTS
  def compare_test():
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

  def verify_test():
    facenet.recognize(HOME + "/PycharmProjects/facial-recognition/images/test_images/ryan.jpg")

  def realtime_recognize_test():
    facenet.realtime_recognize()

  # TESTING
  realtime_recognize_test()
