import keras
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from mtcnn.mtcnn import MTCNN
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
import functools

# CONSTANTS
HOME = os.getenv("HOME")
CASCADE_PATH = HOME + "/PycharmProjects/ai-security/models/haarcascade_frontalface_alt.xml"
IMAGE_DIR_BASEPATH = HOME + "/PycharmProjects/ai-security/images/"
NAMES = os.listdir(IMAGE_DIR_BASEPATH)

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

  @timer(message="Model load time")
  def __init__(self, filepath):
    self.k_model = keras.models.load_model(filepath)
    self.data = None # must be filled in by user

  def calc_dist(self, img_a, img_b):
    assert self.data, "data must be provided"
    return distance.euclidean(self.data[img_a]["emb"], self.data[img_b]["emb"])

  def calc_dist_plot(self, img_a, img_b):
    assert self.data, "data must be provided"

    dist = self.calc_dist(img_a, img_b)
    is_same = dist <= FaceNet.ALPHA

    print("L2 normalized distance: {} -> {} and {} are the same person: {}".format(dist, img_a, img_b, is_same))

    plt.subplot(1, 2, 1)
    plt.imshow(imread(self.data[img_a]["image_filepath"]))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(imread(self.data[img_b]["image_filepath"]))
    plt.axis("off")

    plt.suptitle("Same person: {}\n L2 distance: {}".format(is_same, dist))
    plt.show()

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
  
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y

  @staticmethod
  def l2_normalize(x, axis=-1, epsilon=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

  @staticmethod
  def load_and_align_images(filepaths, margin):
    # cascade = cv2.CascadeClassifier(CASCADE_PATH)
    detector = MTCNN()

    aligned_images = []
    for filepath in filepaths:
      img = imread(filepath)
      # faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
      faces = detector.detect_faces(img)[0]["box"]

      if len(faces) == 0:
        raise ValueError("face was not found in {}".format(filepath))
  
      x, y, w, h = faces
      cropped = img[y - margin // 2:y + h + margin // 2, x - margin // 2:x + w + margin // 2, :]
      aligned = resize(cropped, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE), mode="reflect")
      aligned_images.append(aligned)
  
    return np.array(aligned_images)

  @staticmethod
  def calc_embs(facenet, filepaths, margin=10, batch_size=1):
    aligned_images = Preprocessing.prewhiten(Preprocessing.load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
      pd.append(facenet.k_model.predict_on_batch(aligned_images[start:start + batch_size]))

    return Preprocessing.l2_normalize(np.concatenate(pd))

  @staticmethod
  @timer(message="Data Preprocessinging time")
  def custom_load(facenet):
    data = {}
    for name in NAMES:
      image_dirpath = IMAGE_DIR_BASEPATH + name
      image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
      embs = Preprocessing.calc_embs(facenet, image_filepaths)
      for i in range(len(image_filepaths)):
        data["{}{}".format(name, i)] = {"image_filepath": image_filepaths[i], "emb": embs[i]}
    return data

if __name__ == "__main__":
  facenet = FaceNet(HOME + "/PycharmProjects/ai-security/models/facenet_keras.h5")
  facenet.data = Preprocessing.custom_load(facenet)

  start = time()

  # facenet.calc_dist_plot("trump0", "trump1")
  # facenet.calc_dist_plot("trump1", "obama1")
  # facenet.calc_dist_plot("obama0", "obama1")
  #
  # facenet.calc_dist_plot("moon_jae_in0", "moon_jae_in1")
  # facenet.calc_dist_plot("andrew_ng1", "moon_jae_in0")
  # facenet.calc_dist_plot("andrew_ng0", "andrew_ng1")

  facenet.calc_dist_plot("ryan0", "ryan2")
  facenet.calc_dist_plot("ryan1", "ryan2")
  facenet.calc_dist_plot("ryan1", "ryan0")

  facenet.calc_dist_plot("ryan1", "andrew_ng1")
  facenet.calc_dist_plot("ryan0", "andrew_ng0")

  print("Average time per comparison: {}s".format(round((time() - start) / 3, 3)))