"""Facial recognition with FaceNet in Keras, TensorFlow, or TensorRT.
"""

import json
import os
from timeit import default_timer as timer

import cv2
import numpy as np
from sklearn import neighbors, svm
from termcolor import colored

from util.detection import FaceDetector
from util.distance import DistMetric
from util.engine import CudaEngine
from util.loader import (print_time, screen_data, strip_id,
                         retrieve_embeds, get_frozen_graph)
from util.paths import (DB_LOB, DEFAULT_MODEL, CONFIG_HOME,
                        EMBED_KEY_PATH, NAME_KEY_PATH)
from util.visuals import Camera, GraphicsRenderer


class FaceNet:
    """Class implementation of FaceNet"""
    MODELS = json.load(open(CONFIG_HOME + "/defaults/models.json",
                            encoding="utf-8"))

    @print_time("model load time")
    def __init__(self, model_path=DEFAULT_MODEL, data_path=DB_LOB,
                 input_name=None, output_name=None, input_shape=None,
                 classifier="knn", allow_gpu_growth=False):
        """Initializes FaceNet object
        :param model_path: path to model (default: utils.paths.DEFAULT_MODEL)
        :param data_path: path to data (default: utils.paths.DB_LOB)
        :param input_name: name of input tensor (default: None)
        :param output_name: name of output tensor (default: None)
        :param input_shape: input shape (default: None)
        :param classifier: classifier type (default: 'knn')
        :param allow_gpu_growth: allow GPU growth (default: False)
        """

        assert os.path.exists(model_path), f"{model_path} not found"
        assert not data_path or os.path.exists(data_path), \
            f"{data_path} not found"

        if allow_gpu_growth:
            import tensorflow.compat.v1 as tf  # noqa
            options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=options)
            self.sess = tf.Session(config=config)
            self.sess.__enter__()

        if ".h5" in model_path:
            self._keras_init(model_path)
        elif ".tflite" in model_path:
            self._tflite_init(model_path)
        elif ".pb" in model_path:
            self._tf_init(model_path, input_name, output_name, input_shape)
        elif ".engine" in model_path:
            self._trt_init(model_path, input_name, output_name, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")

        self._db = {}
        self.classifier = None
        self.classifier_type = classifier

        if data_path:
            self.set_data(*retrieve_embeds(data_path,
                                           name_keys=NAME_KEY_PATH,
                                           embedding_keys=EMBED_KEY_PATH))
        else:
            print("[DEBUG] data not set. Set it manually with set_data")

    @property
    def data(self):
        """Property for static database
        :returns: self._db
        """

        return self._db

    @property
    def metadata(self):
        return {"metric": self.dist_metric.metric,
                "normalize": self.dist_metric.normalize,
                "alpha": self.alpha,
                "img_norm": self.img_norm}

    def _keras_init(self, filepath):
        """Initializes a Keras model
        :param filepath: path to model (.h5)
        """

        import tensorflow.compat.v1 as tf  # noqa
        self.MODE = "keras"
        self.facenet = tf.keras.models.load_model(filepath)
        self.img_shape = self.facenet.input_shape[1:3]

    def _tflite_init(self, filepath):
        """Initializes a tflite model interpreter
        :param filepath: path to model (.tflite)
        """

        import tensorflow.compat.v1 as tf  # noqa
        self.MODE = "tflite"
        self.facenet = tf.lite.Interpreter(model_path=filepath)
        self.facenet.allocate_tensors()

        self.input_details = self.facenet.get_input_details()
        self.output_details = self.facenet.get_output_details()
        self.img_shape = self.input_details[0]["shape"].tolist()[1:-1]

    def _tf_init(self, filepath, input_name, output_name, input_shape):
        """Initializes a TensorFlow model
        :param filepath: path to model (.pb)
        :param input_name: name of input tensor
        :param output_name: name of output tensor
        :param input_shape: input shape for facenet
        """

        import tensorflow.compat.v1 as tf  # noqa
        self.MODE = "tf"

        graph_def = get_frozen_graph(filepath)
        self.sess = tf.keras.backend.get_session()

        tf.import_graph_def(graph_def, name="")

        self._tensor_init(model_name=filepath,
                          input_name=input_name,
                          output_name=output_name)
        self.facenet = self.sess.graph

        try:
            if input_shape is not None:
                assert input_shape[-1] == 3, \
                    "input shape must be channels-last for tensorflow mode"
                self.img_shape = input_shape[:-1]
            else:
                input_tensor = self.facenet.get_tensor_by_name(
                        self.input_name)
                self.img_shape = input_tensor.get_shape().as_list()[1:3]

        except ValueError:
            self.img_shape = (160, 160)
            print(f"[DEBUG] using default size of {self.img_shape}")

    def _tensor_init(self, model_name, input_name, output_name):
        """Initializes tensors (TF or TRT modes only)
        :param model_name: name of model
        :param input_name: input tensor name
        :param output_name: output tensor name
        """

        self.model_config = self.MODELS["_default"]

        for model in self.MODELS:
            if model in model_name:
                self.model_config = self.MODELS[model]

        self.input_name = self.model_config["input"]
        self.output_name = self.model_config["output"]

        if not self.input_name:
            self.input_name = input_name
        elif not self.output_name:
            self.output_name = output_name

        assert self.input_name and self.output_name, \
            f"I/O tensors for {model_name} not detected or provided"

    def _trt_init(self, filepath, input_name, output_name, input_shape):
        """TensorRT initialization
        :param filepath: path to serialized engine
        :param input_name: name of input to network
        :param output_name: name of output to network
        :param input_shape: input shape (channels first)
        """

        self.MODE = "trt"
        try:
            self.facenet = CudaEngine(filepath, input_name,
                                      output_name, input_shape)
        except NameError:
            raise ValueError("trt mode not available")
        self.img_shape = list(reversed(self.facenet.input_shape))[:-1]

    def add_entry(self, person, embeddings, train_classifier=True):
        """Adds entry (person, embeddings) to database
        :param person: new entry
        :param embeddings: new entry's list of embeddings
        :param train_classifier: train classifier (default: True)
        """
        screen_data(person, embeddings)

        embeds = np.array(embeddings).reshape(len(embeddings), -1)
        self._db[person] = embeds

        stripped = strip_id(person)
        self._stripped_names.append(stripped)

        try:
            embeds = np.concatenate([self._stripped_db[stripped], embeds])
        except KeyError:
            pass
        self._stripped_db[stripped] = embeds

        if train_classifier:
            self._train_classifier()

    def remove_entry(self, person, train_classifier=True):
        """Removes all embeds of person from database.
        :param person: entry to remove
        :param train_classifier: train classifier (default: True)
        """
        keys = list(self.data.keys())
        stripped = strip_id(person)

        for name in keys:
            if strip_id(name) == stripped:
                del self._db[name]
                del self._stripped_names[keys.index(name)]

                try:
                    del self._stripped_db[stripped]
                except KeyError:
                    pass

        if train_classifier:
            self._train_classifier()

    def set_data(self, data, metadata):
        """Sets data property
        :param data: new data in form {name: embedding vector, ...}
        :param metadata: data metadata
        """
        assert metadata, "metadata must be provided"

        self._db = {}
        self._stripped_db = {}
        self._stripped_names = []
        self.data_cfg = metadata

        self.dist_metric = DistMetric(self.data_cfg["metric"],
                                      self.data_cfg["normalize"],
                                      self.data_cfg.get("mean"))
        self.alpha = self.data_cfg["alpha"]
        self.img_norm = self.data_cfg["img_norm"]

        if data:
            for person, embed in data.items():
                self.add_entry(person, embed, train_classifier=False)
            self._train_classifier()

    def _train_classifier(self):
        """Trains person classifier"""
        try:
            embeds = np.squeeze(list(self.data.values()), axis=1)

            if self.classifier_type == "svm":
                self.classifier = svm.SVC(kernel="linear")
                self.classifier.fit(embeds, self._stripped_names)

            elif self.classifier_type == "knn":
                self.classifier = neighbors.NearestNeighbors(
                    radius=self.alpha,
                    metric=self.dist_metric.metric,
                    n_jobs=-1)
                self.classifier.fit(embeds)

        except (AttributeError, ValueError):
            raise ValueError("Current model incompatible with database")

    def normalize(self, imgs):
        if self.img_norm == "per_image":
            # linearly scales x to have mean of 0, variance of 1
            std_adj = np.std(imgs, axis=(1, 2, 3), keepdims=True)
            std_adj = np.maximum(std_adj, 1. / np.sqrt(imgs.size / len(imgs)))
            mean = np.mean(imgs, axis=(1, 2, 3), keepdims=True)
            return (imgs - mean) / std_adj
        elif self.img_norm == "fixed":
            # scales x to [-1, 1]
            return (imgs - 127.5) / 128.

    def embed(self, imgs):
        """Embeds cropped face
        :param imgs: list of cropped faces with shape (b, h, w, 3)
        :returns: embedding as array with shape (1, -1)
        """

        if self.MODE == "keras":
            embeds = self.facenet.predict(imgs, batch_size=len(imgs))  # noqa
        elif self.MODE == "tf":
            out = self.facenet.get_tensor_by_name(self.output_name)  # noqa
            embeds = self.sess.run(out, feed_dict={self.input_name: imgs})
        elif self.MODE == "tflite":
            imgs = imgs.astype(np.float32)
            self.facenet.set_tensor(self.input_details[0]["index"], imgs)
            self.facenet.invoke()
            embeds = self.facenet.get_tensor(self.output_details[0]["index"])
        else:
            embeds = self.facenet.inference(imgs)
        return embeds.reshape(len(imgs), -1)

    def predict(self, img, detector, margin=10, flip=False, verbose=True):
        """Embeds and normalizes an image from path or array
        :param img: image to be predicted on (BGR image)
        :param detector: FaceDetector object
        :param margin: margin for MTCNN face cropping (default: 10)
        :param flip: flip and concatenate or not (default: False)
        :param verbose: verbosity (default: True)
        :returns: normalized embeddings, facial coordinates
        """

        cropped_faces, face_coords = detector.crop_face(img, margin,
                                                        flip, verbose)
        assert cropped_faces is not None, "no face detected"

        start = timer()

        normalized = self.normalize(np.array(cropped_faces))
        embeds = self.embed(normalized)
        embeds = self.dist_metric.apply_norms(embeds, batch=True)

        if verbose:
            elapsed = round(1000. * (timer() - start), 2)
            time = colored(f"{elapsed} ms", attrs=["bold"])
            vecs = f"{len(embeds)} vector{'s' if len(embeds) > 1 else ''}"
            print(f"Embedding time ({vecs}): " + time)

        return embeds, face_coords

    def recognize(self, img, *args, eps=2.0, verbose=True, **kwargs):
        """Facial recognition
        :param img: image array in BGR mode
        :param eps: controls inv dist sensitivity (default: 2.0)
        :param verbose: verbose or not (default: True)
        :param args: will be passed to self.predict
        :param kwargs: will be passed to self.predict
        :returns: face, is recognized, best match, time elapsed
        """
        start = timer()

        is_recognized = None
        best_match = None
        face = None

        try:
            if self.classifier_type == "svm":
                embeds, face = self.predict(img, *args, **kwargs)
                best_match = self.classifier.predict(embeds)[0]

                nearest = self._stripped_db[best_match]
                dists = self.dist_metric.distance(embeds, nearest, True)
                dist = np.average(dists)

                is_recognized = dist <= self.alpha

            elif self.classifier_type == "knn":
                embeds, face = self.predict(img, *args, **kwargs)
                dists, idxs = self.classifier.radius_neighbors(embeds)
                dists = dists.flatten()[0]
                idxs = idxs.flatten()[0]

                if len(idxs) != 0:
                    matches = np.take(self._stripped_names, idxs).tolist()
                    key = lambda i: matches.count(matches[i]) + eps/dists[i]
                    best_idx = max(range(len(matches)), key=key)

                    best_match, dist = matches[best_idx], dists[best_idx]
                    is_recognized = dist <= self.alpha

            if verbose:
                info = colored(f"{round(dist, 4)} ({best_match})",
                               color="green" if is_recognized else "red")
                print(f"{self.dist_metric}: {info}")

        except (ValueError, AssertionError, cv2.error) as error:
            incompatible = "query data dimension"
            if isinstance(error, ValueError) and incompatible in str(error):
                raise ValueError("Current model incompatible with database")
            elif isinstance(error, cv2.error) and "resize" in str(error):
                print("Frame capture failed")
            elif "no face detected" not in str(error):
                raise error

        elapsed = round(1000. * (timer() - start), 4)
        return face, is_recognized, best_match, elapsed

    def real_time_recognize(self, width=640, height=360, resize=1.,
                            detector="mtcnn", flip=False, graphics=True, socket=None):
        """Real-time facial recognition
        :param width: width of frame (default: 640)
        :param height: height of frame (default: 360)
        :param resize: resize scale (default: 1. = no resize)
        :param detector: face detector type (default: "mtcnn")
        :param flip: whether to flip horizontally or not (default: False)
        :param graphics: whether or not to use graphics (default: True)
        """

        assert self._db, "data must be provided"
        assert 0. <= resize <= 1., "resize must be in [0., 1.]"

        graphics_controller = GraphicsRenderer(width, height, resize)
        cap = Camera(width, height)

        detector = FaceDetector(detector, self.img_shape, min_face_size=240)

        while True:
            _, frame = cap.read()
            cframe = frame.copy()

            # resize frame
            if resize != 1:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            # facial detection and recognition
            info = self.recognize(frame, detector, flip=flip)

            if socket: 
                socket.send(json.dumps({"best_match":info[2]}))

            # graphics
            if graphics:
                graphics_controller.add_graphics(cframe, *info)
                cv2.imshow("AI Security v2021.0.1", cframe)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

