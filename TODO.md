# To-do

Ordered in terms of precedence. Used internally instead of Issues.

## Development

### [x] Optimize pre-processing

Current preprocessing code (see `Preprocessing` class in facenet.py) is really slow; as a result, on an NVIDIA RTX 2060 
GPU (a pretty powerful GPU), classification of a single image takes around 1.5 seconds-- way too slow for production 
code.

### [x] Integrate real_time_face.py into `FaceNet` class
Ideally, we would be able to run

```python

from aisecurity.facenet import FaceNet, Preprocessing

facenet = FaceNet("/path/to/model")
image_dir = "/path/to/images"
people = ["some", "list", "of", "people"]
facenet.set_data(Preprocessing.load(facenet.get_facenet(), image_dir, people))

facenet.real_time_recognize()
# real-time facial recognition, fast and accurate enough to be used in production

```

### [ ] Convert TensorFlow .pb model to .h5 file 

We're currently using a pretrained model from https://github.com/nyoki-mtl/keras-facenet. This model was trained on MS-CELEB-1M and achieved 99.4% LFW accuracy.

However, this repository (https://github.com/davidsandberg/facenet) has a model with the same architecture trained on VGGFace2 (3 million images) that achieved 99.65% LFW accuracy. It is therefore preferable to the Keras FaceNet that we're currently using. The problem with using this model is that it is provided as .pb and .meta files, which are compatible with TensorFlow but not Keras.

It would be great if someone could convert the TensorFlow FaceNet model into a Keras model (stored as a .h5 file).

TensorFlow model: https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view

### - [ ] Database appearance log

[x] Log appearances of people and the corresponding time in a database during real time facial recognition. 

[x] Make log code object-oriented (i.e., integrate into FaceNet class as with the K-NN code). Fill in the not implemented functions (`FaceNet._log_init`, `FaceNet.log`) and use them in `FaceNet.real_time_recognize`.

[ ] Update logging threshold (@orangese).

### [x] Encryption security with database (@orangese)

AES encryption for JSON image database. 

### [ ] Packaging everything (@orangese)

[x] Make everything installable and configurable with `pip` (@orangese).

[ ] Fix logging in package implementation.

### [ ] Real-time dynamic database update

Add new database that contains a real-time log of unknown people who just signed in. This database can also be used for recognition.

### [ ] Real-time static database update (?)

Constant update of static database using pictures that are being taken real-time. Would probably increase accuracy but might be an issue in terms of privacy.