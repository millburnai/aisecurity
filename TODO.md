# To-do

## Development

### - [ ] Optimize pre-processing

Current preprocessing code (see `Preprocessing` class in facenet.py) is really slow; as a result, on an NVIDIA RTX 2060 
GPU (a pretty powerful GPU), classification of a single image takes around 1.5 seconds-- way too slow for production 
code.

### - [x] Integrate real_time_face.py into `FaceNet` class
Ideally, we would be able to run

```python

from facenet import FaceNet, Preprocessing

facenet = FaceNet("/path/to/model")
image_dir = "/path/to/images"
people = ["some", "list", "of", "people"]
facenet.set_data(Preprocessing.load(facenet.get_facenet(), image_dir, people))

facenet.real_time_recognize()
# real-time facial recognition, fast and accurate enough to be used in production

```

### - [ ] Convert TensorFlow .pb model to .h5 file

See issues for more detail. Our current model works well but has trouble classifying certain races. 