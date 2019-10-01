
"""

"facenet.py"

Facial recognition with FaceNet in Keras.

Paper: https://arxiv.org/pdf/1503.03832.pdf

"""

import warnings
import asyncio
import json
import os
import time
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

from paths import Paths
from encryptions import DataEncryption
import log

# DECORATORS
def timer(message="Time elapsed"):

  def _timer(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
      start = time.time()
      result = func(*args, **kwargs)
      print("{}: {}s".format(message, round(time.time() - start, 3)))
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
    self._data = None # must be filled in by user

  @staticmethod
  def _log_init():
    log.init()
    if log.suspicious is None:
      log.suspicious = log.get_now(True)

  # MUTATORS
  def set_data(self, data):
    assert data is not None, "data must be provided"

    def check_validity(data):
      for key in data.keys():
        assert isinstance(key, str), "data keys must be person names"
        is_vector = data[key].ndim < 2 or (1 in data[key].shape)
        assert isinstance(data[key], np.ndarray) and is_vector, "each data[key] must be a vectorized embedding"
      return data

    self._data = check_validity(data)
    self._set_knn()

  def _set_knn(self):
    k_nn_label_dict, embeddings = [], []
    for person in self._data.keys():
      k_nn_label_dict.append(person[:-1])
      embeddings.append(self._data[person])
    self.k_nn = neighbors.KNeighborsClassifier(n_neighbors=len(k_nn_label_dict) // len(set(k_nn_label_dict)))
    self.k_nn.fit(embeddings, k_nn_label_dict)

  # RETRIEVERS
  @property
  def data(self):
    return self._data

  def get_facenet(self):
    return self.k_model

  def get_embeds(self, *args, **kwargs):
    embeds = []
    for n in args:
      if isinstance(n, str):
        try:
          n = self._data[n]
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
    assert self._data, "data must be provided"

    dist = self.l2_dist(a, b)
    is_same = dist <= FaceNet.ALPHA

    if verbose:
      print("L2 distance: {} -> {} and {} are the same person: {}".format(dist, a, b, is_same))

    return int(is_same), dist

  # FACIAL RECOGNITION HELPER
  @timer(message="Recognition time")
  def _recognize(self, img, faces=None, margin=15):
    assert self._data, "data must be provided"

    embedding = self.get_embeds(img, faces=faces, margin=margin)
    best_match = self.k_nn.predict(embedding)[0]

    l2_dist = self.l2_dist(embedding, self._data[best_match + "0"])

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

    return is_recognized, best_match, l2_dist

  # REAL-TIME FACIAL RECOGNITION HELPER
  async def _real_time_recognize(self, width, height, use_log):
    # TODO(22pilarskil): fill in code templates
    if use_log:
      self._log_init()

    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    line_thickness = round(1e-6 * width * height + 1.5)
    radius = round((1e-6 * width * height + 1.5) / 2.)
    font_size = 4.5e-7 * width * height + 0.5
    # works for 6.25e4 pixel video cature to 1e6 pixel video capture
    # TODO: make font_size more adaptive (use cv2.getTextSize())

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

          if use_log:
            self.log_activity(is_recognized, best_match, frame)
          
          corner = (x - self.MARGIN // 2, y - self.MARGIN // 2)
          box = (x + height + self.MARGIN // 2, y + width + self.MARGIN // 2)

          FaceNet.add_key_points(overlay, key_points, radius, color, line_thickness)
          cv2.addWeighted(overlay, 1.0 - self.TRANSPARENCY, frame, self.TRANSPARENCY, 0, frame)

          FaceNet.add_box_and_label(frame, corner, box, color, line_thickness, best_match, font_size, thickness=1)

      else:
        print("No face detected")

      cv2.imshow("CSII AI facial recognition v0.1", frame)

      await asyncio.sleep(K.epsilon())

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    cv2.destroyAllWindows()

  # REAL-TIME FACIAL RECOGNITION
  def real_time_recognize(self, width=500, height=250, use_log=True):

    async def async_helper(recognize_func, *args, **kwargs):
      await recognize_func(*args, **kwargs)

    loop = asyncio.new_event_loop()
    task = loop.create_task(async_helper(self._real_time_recognize, width=width, height=height, use_log=use_log))
    loop.run_until_complete(task)

  # DISPLAYING
  @staticmethod
  def add_box_and_label(frame, corner, box, color, line_thickness, best_match, font_size, thickness):
    cv2.rectangle(frame, corner, box, color, line_thickness)
    cv2.putText(frame, best_match, corner, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)

  @staticmethod
  def add_key_points(overlay, key_points, radius, color, line_thickness):
    cv2.circle(overlay, (key_points["left_eye"]), radius, color, line_thickness)
    cv2.circle(overlay, (key_points["right_eye"]), radius, color, line_thickness)
    cv2.circle(overlay, (key_points["nose"]), radius, color, line_thickness)
    cv2.circle(overlay, (key_points["mouth_left"]), radius, color, line_thickness)
    cv2.circle(overlay, (key_points["mouth_right"]), radius, color, line_thickness)

    cv2.line(overlay, key_points["left_eye"], key_points["nose"], color, radius)
    cv2.line(overlay, key_points["right_eye"], key_points["nose"], color, radius)
    cv2.line(overlay, key_points["mouth_left"], key_points["nose"], color, radius)
    cv2.line(overlay, key_points["mouth_right"], key_points["nose"], color, radius)

  def show_embeds(self, encrypted=False, single=False):

    def closest_multiples(n):
      if n == 0 or n == 1: return n, n
      factors = [((i, int(n / i)), (abs(i - int(n / i)))) for i in range(1, n) if n % i == 0]
      return factors[np.argmin(list(zip(*factors))[1]).item()][0]

    data = DataEncryption.encrypt_data(self.data, ignore=["embeddings"], decryptable=False) if encrypted else self.data
    for person in data:
      embed = np.asarray(data[person])
      embed = embed.reshape(*closest_multiples(embed.shape[0]))

      plt.imshow(embed, cmap="gray")
      try:
        plt.title(person)
      except TypeError:
        warnings.warn("encrypted data cannot be displayed due to presence of non-UTF8-decodable values")
      plt.axis("off")
      plt.show()

      if single and person == list(data.keys())[0]:
        break

  # LOGGING
  @staticmethod
  def log_activity(is_recognized, best_match, frame):
    get_path = lambda num: Paths.HOME + "/images/_suspicious/{}.jpg".format(num)

    log.rec_threshold = log.update_rec_threshold(is_recognized)
    log.unrec_threshold = log.update_unrec_threshold(is_recognized)

    if log.unrec_threshold > log.THRESHOLD and (log.get_now(True) - log.suspicious).total_seconds() > log.THRESHOLD:
      path = get_path(log.num_suspicious)
      cv2.imwrite(path, frame)
      log.add_suspicious(path)
      print("Suspicious activity")

    if log.rec_threshold > log.THRESHOLD and log.verify_repeat(best_match):
      log.add_transaction(best_match)
      print("New transaction recorded")

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
  def load(facenet, img_dir):
    people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
    data = {}
    for person in people:
      person_dir = img_dir + person
      img_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if not f.endswith(".DS_Store")]
      embeddings = Preprocessing.embed(facenet, img_paths)
      for index, path in enumerate(img_paths):
        data["{}{}".format(person, index)] = embeddings[index]
    return data

  @staticmethod
  @timer(message="Data dumping time")
  def dump_embeds(facenet, path, img_dir, overwrite=False):
    people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
    if not overwrite:
      old_embeds = Preprocessing.retrieve_embeds(path)
      new_people = [person for person in people if person + "0" not in old_embeds.keys()]
      new_embeds = Preprocessing.load(facenet.get_facenet(), img_dir, new_people)
      embeds_dict = {**old_embeds, **new_embeds} # combining dicts and overwriting any duplicates with new_embeds
    else:
      embeds_dict = Preprocessing.load(facenet.get_facenet(), img_dir, people)

    encrypted_data = DataEncryption.encrypt_data(embeds_dict)

    with open(path, "w+") as json_file:
      json.dump(encrypted_data, json_file, indent=4, sort_keys=True, ensure_ascii=False)

  @staticmethod
  @timer(message="Data retrieval time")
  def retrieve_embeds(path, encrypted=True):
    with open(path, "r") as json_file:
      data = json.load(json_file)
    return DataEncryption.decrypt_data(data) if encrypted else data

# TESTS
class Tests(object):

  @staticmethod
  def redump():
    data = Preprocessing.retrieve_embeds(Paths.HOME + "/images/_processed.json", False)
    with open("/images/encrypted.json", "w") as json_file:
      json.dump(DataEncryption.encrypt_data(data), json_file, indent=4)
    data = Preprocessing.retrieve_embeds(Paths.HOME + "/images/encrypted.json")
    print(list(data.keys()))

  @staticmethod
  def compare_test(facenet):
    start = time.time()

    my_imgs = []
    for person in Paths.HOME:
      for index in range(len([f for f in os.listdir(Paths.img_dir + person) if not f.endswith(".DS_Store")])):
        my_imgs.append("{}{}".format(person, index))

    count = 0
    for img_a in my_imgs:
      for img_b in my_imgs:
        if not np.array_equal(img_a, img_b):
          facenet.compare(img_a, img_b)
          count += 1

    print("Average time per comparison: {}s".format(round((time.time() - start) / count, 3)))

  @staticmethod
  def recognize_test(facenet):
    facenet.recognize(Paths.HOME + "/images/_test_images/ryan.jpg")

  @staticmethod
  async def real_time_recognize_test(facenet, use_log=True):
    await facenet.real_time_recognize(use_log=use_log)

if __name__ == "__main__":
  facenet = FaceNet(Paths.HOME + "/models/facenet_keras.h5")
  facenet.set_data(Preprocessing.retrieve_embeds(Paths.HOME + "/images/encrypted.json"))
