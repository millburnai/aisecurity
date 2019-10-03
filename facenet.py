
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
from termcolor import cprint

import matplotlib.pyplot as plt
import keras
from keras import backend as K
import numpy as np
import cv2
from skimage.transform import resize
from sklearn import neighbors
from imageio import imread
from mtcnn.mtcnn import MTCNN

from extras.paths import Paths
from security.encryptions import DataEncryption
from logs import log

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
    if use_log:
      log.init(flush=True, thresholds={"max_error": self.ALPHA})

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

          corner = (x - self.MARGIN // 2, y - self.MARGIN // 2)
          box = (x + height + self.MARGIN // 2, y + width + self.MARGIN // 2)

          FaceNet.add_key_points(overlay, key_points, radius, color, line_thickness)
          cv2.addWeighted(overlay, 1.0 - self.TRANSPARENCY, frame, self.TRANSPARENCY, 0, frame)

          FaceNet.add_box_and_label(frame, corner, box, color, line_thickness, best_match, font_size, thickness=1)

          if use_log:
            self.log_activity(is_recognized, best_match, frame, l2_dist, log_susp=True)

      else:
        log.flush_current()
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
  def log_activity(is_recognized, best_match, frame, l2_dist, log_susp=True):
    cooldown_ok = lambda t: time.time() - t > log.THRESHOLDS["cooldown"]

    def get_mode(d): #gets highest number in current log
      max_key = list(d.keys())[0]
      for key in d:
        if len(d[key]) > len(d[max_key]):
          max_key = key
      return max_key

    log.update_current_logs(is_recognized, best_match, l2_dist)

    if log.num_unrecognized >= log.THRESHOLDS["num_unrecognized"] and cooldown_ok(log.unrec_last_logged) and log_susp:
      path = Paths.HOME + "/images/_suspicious/{}.jpg".format(len(os.listdir(Paths.HOME + "/images/_suspicious")))
      cv2.imwrite(path, frame)
      log.log_suspicious(path)
      cprint("Suspicious activity logged", color="red", attrs=["bold"])

    if log.num_recognized >= log.THRESHOLDS["num_recognized"] and cooldown_ok(log.rec_last_logged):
      if log.get_percent_diff(best_match) <= log.THRESHOLDS["percent_diff"]:
        recognized_person = get_mode(log.current_log)
        log.log_person(recognized_person, times=log.current_log[recognized_person])
        cprint("Regular activity logged", color="green", attrs=["bold"])

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
  def load(facenet, img_dir, people=None):
    if people is None:
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
  def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, overwrite=False):
    people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
    if not overwrite:
      old_embeds = Preprocessing.retrieve_embeds(retrieve_path if retrieve_path is not None else dump_path)
      new_people = [person for person in people if person + "0" not in old_embeds.keys()]
      new_embeds = Preprocessing.load(facenet.get_facenet(), img_dir, people=new_people)
      embeds_dict = {**old_embeds, **new_embeds} # combining dicts and overwriting any duplicates with new_embeds
    else:
      embeds_dict = Preprocessing.load(facenet.get_facenet(), img_dir, people)

    encrypted_data = DataEncryption.encrypt_data(embeds_dict)

    with open(dump_path, "w+") as json_file:
      json.dump(encrypted_data, json_file, indent=4, ensure_ascii=False)

  @staticmethod
  @timer(message="Data retrieval time")
  def retrieve_embeds(path, encrypted=True):
    with open(path, "r") as json_file:
      data = json.load(json_file)
    return DataEncryption.decrypt_data(data) if encrypted else data
