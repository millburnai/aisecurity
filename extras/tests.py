
"""

"tests.py"

Unit testing for "facenet.py".

"""

from facenet import *

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

def recognize_test(facenet):
  facenet.recognize(Paths.HOME + "/images/_test_images/ryan.jpg")

async def real_time_recognize_test(facenet, use_log=True):
  await facenet.real_time_recognize(use_log=use_log)

if __name__ == "__main__":
  print("Nothing yet!")