
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
  MARGIN = 10
  TRANSPARENCY = 0.5

  # INITS
  @timer(message="Model load time")
  def __init__(self, filepath):
    self.k_model = keras.models.load_model(filepath)
    self.data = None # must be filled in by user

  # MUTATORS AND RETRIEVERS
  def set_data(self, data):
    assert data is not None, "data must be provided"
    self.data = data
    self._set_knn()

  def _set_knn(self):
    k_nn_label_dict, embeddings = [], []
    for person in self.data.keys():
      k_nn_label_dict.append(person[:-1])
      embeddings.append(self.data[person]["embedding"])
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
        except KeyError:
          n = self.predict([n], **kwargs)
      elif not (n.ndim < 2 or (1 in n.shape)):
        n = self.predict([n], **kwargs)
      embeds.append(n)
    return tuple(embeds) if len(embeds) > 1 else embeds[0]

  # LOW-LEVEL COMPARISON FUNCTIONS
  def l2_dist(self, a, b):
    a, b = self.get_embeds(a, b)
    return np.linalg.norm(a - b)

  def predict(self, paths_or_imgs, batch_size=1, faces=None, margin=10):
    return Preprocessing.embed(self.k_model, paths_or_imgs, batch_size=batch_size, faces=faces, margin=margin)

  # FACIAL COMPARISON
  def compare(self, a, b, verbose=True):
    assert self.data, "data must be provided"

    dist = self.l2_dist(a, b)
    is_same = dist <= FaceNet.ALPHA

    if verbose:
      print("L2 distance: {} -> {} and {} are the same person: {}".format(dist, a, b, is_same))
      self.disp_imgs(a, b, title = "Same person: {}\n L2 distance: {}".format(is_same, dist))

    return int(is_same), dist

  # FACIAL RECOGNITION HELPER
  @timer(message="Recognition time")
  def _recognize(self, img, faces=None, margin=15):
    assert self.data, "data must be provided"

    embedding = self.get_embeds(img, faces=faces, margin=margin)
    best_match = self.k_nn.predict(embedding)[0]

    l2_dist = self.l2_dist(embedding, self.data[best_match + "0"]["embedding"])

    return int(l2_dist <= FaceNet.ALPHA), best_match, l2_dist

  # FACIAL RECOGNITION
  def recognize(self, img, verbose=True):
    # img can be a path, image, database name, or embedding
    is_recognized, best_match, l2_dist = self._recognize(img)

    if verbose:
      if is_recognized:
        print("Your image is a picture of \"{}\": L2 distance of {}".format(best_match, l2_dist))
      else:
        print("Your image is not in the database. The best match is \"{}\" with an L2 distance of ".format(
          best_match, l2_dist))
        self.disp_imgs(img, "{}0".format(best_match), title="Best match: {}\nL2 distance: {}".format(
          best_match, l2_dist))

    return is_recognized, best_match, l2_dist

  # REAL TIME RECOGNITION
  async def real_time_recognize(self, width=500, height=250):
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    line_thickness = round((width + height) / 375.)
    radius = round(line_thickness / 2.)

    while True:
      _, frame = cap.read()
      result = detector.detect_faces(frame)

      if result:
        overlay = frame.copy()

        for person in result:
          face = person["box"]
          key_points = person["keypoints"]
          x, y, height, width = face

          try:
            is_recognized, best_match, l2_dist = self._recognize(frame, faces=face, margin=self.MARGIN)
            print("L2 distance: {} ({})".format(l2_dist, best_match))
          except ValueError:
            print("Image refresh rate too high")
            continue

          color = (0, 255, 0) if is_recognized else (0, 0, 255) # green if is_recognized else red

          corner = (x - self.MARGIN // 2, y - self.MARGIN // 2)
          box = (x + height + self.MARGIN // 2, y + width + self.MARGIN // 2)

          FaceNet.add_box_and_label(frame, corner, box, color, radius, line_thickness, best_match)
          FaceNet.add_key_points(overlay, key_points, radius, color, line_thickness)

        cv2.addWeighted(overlay, 1.0 - self.TRANSPARENCY, frame, 1.0, 0, frame)

        await asyncio.sleep(K.epsilon())

      else:
        print("No face detected")
        await asyncio.sleep(K.epsilon())

      cv2.imshow("CSII AI facial recognition v0.1", frame)

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

  @staticmethod
  def add_box_and_label(frame, corner, box, color, radius, line_thickness, best_match):
    cv2.rectangle(frame, corner, box, color, thickness=line_thickness)
    cv2.putText(frame, best_match, org=corner, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, color=color)

  @staticmethod
  def add_key_points(overlay, key_points, radius, color, line_thickness):
    cv2.circle(overlay, (key_points["left_eye"]), radius=radius, color=color, thickness=line_thickness)
    cv2.circle(overlay, (key_points["right_eye"]), radius=radius, color=color, thickness=line_thickness)
    cv2.circle(overlay, (key_points["nose"]), radius=radius, color=color, thickness=line_thickness)
    cv2.circle(overlay, (key_points["mouth_left"]), radius=radius, color=color, thickness=line_thickness)
    cv2.circle(overlay, (key_points["mouth_right"]), radius=radius, color=color, thickness=line_thickness)

    cv2.line(overlay, key_points["left_eye"], key_points["nose"], color, thickness=radius)
    cv2.line(overlay, key_points["right_eye"], key_points["nose"], color, thickness=radius)
    cv2.line(overlay, key_points["mouth_left"], key_points["nose"], color, thickness=radius)
    cv2.line(overlay, key_points["mouth_right"], key_points["nose"], color, thickness=radius)

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
  def dump_embeds(facenet, path, img_dir, people):
    embeds_dict = Preprocessing.load(facenet.get_facenet(), img_dir, people)
    with open(path, "w+") as json_file:
      json.dump(embeds_dict, json_file)

  @staticmethod
  @timer(message="Data retrieval time")
  def retrieve_embeds(path):
    with open(path, "r") as json_file:
      data = json.load(json_file)
      for person in data.keys():
        data[person]["embedding"] = np.asarray(data[person]["embedding"])
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

  facenet = FaceNet(Tests.HOME + "/PycharmProjects/facial-recognition/models/facenet_keras.h5")
  # Preprocessing.dump_embeds(facenet, Tests.HOME + "/PycharmProjects/facial-recognition/images/database/processed.json")

  facenet.set_data(Preprocessing.retrieve_embeds(
    Tests.HOME + "/PycharmProjects/facial-recognition/images/database/processed.json"))

  loop = asyncio.new_event_loop()
  task = loop.create_task(Tests.real_time_recognize_test(facenet))
  loop.run_until_complete(task)