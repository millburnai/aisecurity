import keras
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance

HOME = os.getenv("HOME")
ALPHA = 1.0

start = time()
facenet = keras.models.load_model(HOME + "/PycharmProjects/ai-security/models/facenet_keras.h5")
print("Time: {}s".format(round(time() - start, 3)))

cascade_path = HOME + "/PycharmProjects/ai-security/models/haarcascade_frontalface_alt2.xml"
image_dir_basepath = HOME + "/PycharmProjects/ai-security/images/"
names = ["trump", "obama", "moon_jae_in", "andrew_ng"]
image_size = 160

def prewhiten(x):
  if x.ndim == 4:
    axis = (1, 2, 3)
    size = x[0].size
  elif x.ndim == 3:
    axis = (0, 1, 2)
    size = x.size
  else:
    raise ValueError('Dimension should be 3 or 4')

  mean = np.mean(x, axis=axis, keepdims=True)
  std = np.std(x, axis=axis, keepdims=True)
  std_adj = np.maximum(std, 1.0 / np.sqrt(size))
  y = (x - mean) / std_adj
  return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
  output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
  return output

def load_and_align_images(filepaths, margin):
  cascade = cv2.CascadeClassifier(cascade_path)

  aligned_images = []
  for filepath in filepaths:
    if filepath.endswith(".DS_STORE"):
      print(filepath)
      continue

    img = imread(filepath)
    faces = cascade.detectMultiScale(img,
                                     scaleFactor=1.1,
                                     minNeighbors=3)
    if len(faces) == 0:
      raise ValueError("face was not found in {}".format(filepath))

    (x, y, w, h) = faces[0]
    cropped = img[y - margin // 2:y + h + margin // 2,
              x - margin // 2:x + w + margin // 2, :]
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    aligned_images.append(aligned)

  return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
  aligned_images = prewhiten(load_and_align_images(filepaths, margin))
  pd = []
  for start in range(0, len(aligned_images), batch_size):
    pd.append(facenet.predict_on_batch(aligned_images[start:start + batch_size]))
  embs = l2_normalize(np.concatenate(pd))

  return embs

def calc_dist(data, img_name0, img_name1):
    return distance.euclidean(data[img_name0]['emb'], data[img_name1]['emb'])

def calc_dist_plot(data, img_name0, img_name1):
    dist = calc_dist(img_name0, img_name1)
    print("L2 normalized distance: {0} -> {1} and {2} are the same person: {3}".format(dist, img_name0, img_name1,
                                                                                       dist <= ALPHA))
    plt.subplot(1, 2, 1)
    plt.imshow(imread(data[img_name0]['image_filepath']))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(imread(data[img_name1]['image_filepath']))
    plt.axis("off")

    plt.suptitle("Same person: {}".format(dist <= ALPHA))
    plt.show()

if __name__ == "__main__":
  data = {}
  for name in names:
    image_dirpath = image_dir_basepath + name
    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
    embs = calc_embs(image_filepaths)
    for i in range(len(image_filepaths)):
      data['{}{}'.format(name, i)] = {'image_filepath': image_filepaths[i], 'emb': embs[i]}

  calc_dist_plot(data, 'trump0', 'trump1')
  calc_dist_plot(data, 'trump1', 'obama1')
  calc_dist_plot(data, 'obama0', 'obama1')

  calc_dist_plot(data, 'moon_jae_in0', 'moon_jae_in1')
  calc_dist_plot(data, 'andrew_ng1', 'moon_jae_in0')
  calc_dist_plot(data, 'andrew_ng0', 'andrew_ng1')