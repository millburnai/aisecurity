
"""

"facenet.py"

Facial recognition with FaceNet in Keras.

Paper: https://arxiv.org/pdf/1503.03832.pdf

"""
import asyncio
import json
import os
from time import time
import functools

import matplotlib.pyplot as plt
import keras
from keras import backend as K
import numpy as np
import cv2
from skimage.transform import resize
from sklearn import neighbors
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
def timer(message="Time elapsed"):

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
  MARGIN = 15

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
    k_nn_label_dict, embeddings = [], []
    for key in self.data.keys():
      k_nn_label_dict.append(key[:-1])
      embeddings.append(self.data[key]["embedding"])
    self.k_nn = neighbors.KNeighborsClassifier(n_neighbors = 1)
    self.k_nn.fit(embeddings, k_nn_label_dict)

  def get_facenet(self):
    return self.k_model

  def get_embeds(self, *args, **kwargs):
    embeds = []
    for n in args:
      if isinstance(n, str):
        try:
          n = self.data[n]["embedding"]
        except TypeError:
          n = self.predict([n], **kwargs)
      elif not (n.ndim < 2 or (1 in n.shape)):
        n = self.predict(n, **kwargs)
      embeds.append(n)
    return tuple(embeds)

  # LOW-LEVLE COMPARISON FUNCTIONS
  def l2_dist(self, a, b):
    a, b = self.get_embeds(a, b)
    return np.linalg.norm(a - b)

  def predict(self, paths_or_imgs, batch_size=1, faces=None, margin=15):
    return Preprocessing.embed(self.k_model, paths_or_imgs, batch_size=batch_size, faces=faces, margin=margin)

  # FACIAL COMPARISON
  def compare(self, a, b, verbose=True):
    assert self.data, "data must be provided"
    a, b = self.get_embeds(a, b)

    dist = self.l2_dist(a, b)
    is_same = dist <= FaceNet.ALPHA

    if verbose:
      print("L2 distance: {} -> {} and {} are the same person: {}".format(dist, a, b, is_same))
      self.disp_imgs(a, b, title = "Same person: {}\n L2 distance: {}".format(is_same, dist))

    return int(is_same), dist

  # FACIAL RECOGNITION HELPER
  @timer(message="Recognition time")
  def _recognize(self, img, verbose=True, faces=None, margin=15):
    assert self.data, "data must be provided"

    if self.k_nn is None:
      self._set_knn()

    embedding = self.predict([img], faces=faces, margin=margin)
    k_nn_preds = self.k_nn.predict(embedding)

    best_match = k_nn_preds[0]

    l2_dist = self.l2_dist(embedding, self.data[best_match + "0"]["embedding"])

    if verbose:
      if l2_dist <= FaceNet.ALPHA:
        print("Your image is a picture of \"{}\": L2 distance of {}".format(best_match, l2_dist))
      else:
        print("Your image is not in the database. The best match is \"{}\" with an L2 distance of ".format(
          best_match, l2_dist))
        self.disp_imgs(img, "{}0".format(best_match), title="Best match: {}\nL2 distance: {}".format(
          best_match, l2_dist))

    return int(l2_dist <= FaceNet.ALPHA), k_nn_preds[0], l2_dist

  # FACIAL RECOGNITION
  def recognize(self, img):
    # img can be a path, image, database name, or embedding
    return self._recognize(self.get_embeds(img), verbose=True, faces=None)

  # REAL TIME RECOGNITION (DEMO)
  async def real_time_recognize(self, width=500, height=250):
    if self.k_nn is None:
      self._set_knn()

    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
      _, frame = cap.read()
      result = detector.detect_faces(frame)

      if not result:
        print("face not detected")
        await asyncio.sleep(0.01)

      if result:
        for person in result:
          faces = person["box"]
          x, y, height, width = faces

          try:
            is_recognized, best_match, l2_dist = self._recognize(frame, verbose=False, faces=faces, margin=self.MARGIN)
            print("L2 distance: {}".format(l2_dist))
          except ValueError:
            # TODO: find way to circumnavigate this error-- image is empty
            print("Empty image")
            continue

          color = (0, 255, 0) if is_recognized else (0, 0, 255) # green if is_recognize else red

          corner = (x - self.MARGIN // 2, y - self.MARGIN // 2)
          box = (x + height + self.MARGIN // 2, y + width + self.MARGIN // 2)

          cv2.rectangle(frame, corner, box, color, thickness=2)
          cv2.putText(frame, best_match,  org=corner, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=color)

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
  def align_imgs(paths_or_imgs, margin, faces=None):
    if not faces:
      detector = MTCNN()

    def align_img(path_or_img, faces=None):
      try:
        img = imread(path_or_img)
      except OSError: # if img is embedding
        img = path_or_img

      if not faces:
        found = detector.detect_faces(img)
        assert len(found) != 0, "face was not found in {}".format(path_or_img)
        faces = found[0]["box"]

      x, y, width, height = faces
      cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
      resized = resize(cropped, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE), mode="reflect")
      return resized

    return np.array([align_img(path_or_img, faces=faces) for path_or_img in paths_or_imgs])

  @staticmethod
  def embed(facenet, paths_or_imgs, margin=15, batch_size=1, faces=None):
    aligned_imgs = Preprocessing.whiten(Preprocessing.align_imgs(paths_or_imgs, margin, faces=faces))
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
        data["{}{}".format(person, index)] = {"path": path, "embedding": list(np.float64(embeddings[index]))}
    return data

  @staticmethod
  @timer(message="Data dumping time")
  def dump_embeds(facenet, file):
    embeds_dict = Preprocessing.load(facenet.get_facenet(), Tests.img_dir, Tests.people)
    with open(file, "w+") as json_file:
      json.dump(embeds_dict, json_file)

  @staticmethod
  @timer(message="Data retrieval time")
  def retrieve_embeds(path):
    with open(path, "r") as json_file:
      data = json.load(json_file)
      for key in data.keys():
        data[key]["embedding"] = np.asarray(data[key]["embedding"])
      return data

# UNIT TESTING
class Tests(object):

  # CONSTANTS
  HOME = os.getenv("HOME")

  # PATHS
  img_dir = HOME + "/PycharmProjects/facial-recognition/images/database/"
  people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]

  @staticmethod
  def compare_test(facenet):
    start = time()

    my_imgs = []
    for person in Tests.people:
      for index in range(len([f for f in os.listdir(Tests.img_dir + person) if not f.endswith(".DS_Store")])):
        my_imgs.append("{}{}".format(person, index))

    count = 0
    for img_a in my_imgs:
      for img_b in my_imgs:
        if not np.array_equal(img_a, img_b):
          facenet.compare(img_a, img_b)
          count += 1

    print("Average time per comparison: {}s".format(round((time() - start) / count, 3)))

  @staticmethod
  def recognize_test(facenet):
    facenet.recognize(Tests.HOME + "/PycharmProjects/facial-recognition/images/test_images/ryan.jpg")

  @staticmethod
  async def real_time_recognize_test(facenet):
    await facenet.real_time_recognize()

if __name__ == "__main__":
  suppress_tf_warnings()

  # NETWORK INIT
  facenet = FaceNet(Tests.HOME + "/PycharmProjects/facial-recognition/models/facenet_keras.h5")
  # Preprocessing.dump_embeds(facenet, Tests.HOME + "/PycharmProjects/facial-recognition/images/database/processed.json")

  facenet.set_data(Preprocessing.retrieve_embeds(
    Tests.HOME + "/PycharmProjects/facial-recognition/images/database/processed.json"))

  # Tests.recognize_test(facenet)

  # TESTING
  loop = asyncio.new_event_loop()
  task = loop.create_task(Tests.real_time_recognize_test(facenet))
  loop.run_until_complete(task)