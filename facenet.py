"""Facial recognition with FaceNet in Keras, TensorFlow, or TensorRT.
"""

import json
import os
from timeit import default_timer as timer

import cv2
import numpy as np
from sklearn import neighbors
import tensorflow.compat.v1 as tf  # noqa
from termcolor import colored

from dataflow.loader import (print_time, screen_data, strip_id,
                             retrieve_embeds, get_frozen_graph)
from db.log import IntegratedLogger, Logger
from db.connection import Websocket
from optim.engine import CudaEngine
from util.lcd import IntegratedLCDProgressBar
from util.distance import DistMetric
from util.paths import (DB_LOB, DEFAULT_MODEL, CONFIG_HOME,
                         EMBED_KEY_PATH, NAME_KEY_PATH)
from util.visuals import Camera, GraphicsRenderer
from face.detection import FaceDetector


class FaceNet:
    """Class implementation of FaceNet"""
    MODELS = json.load(open(CONFIG_HOME + "/defaults/models.json",
                            encoding="utf-8"))

    @print_time("model load time")
    def __init__(self, model_path=DEFAULT_MODEL, data_path=DB_LOB,
                 input_name=None, output_name=None, input_shape=None,
                 allow_gpu_growth=False):
        """Initializes FaceNet object
        :param model_path: path to model (default: utils.paths.DEFAULT_MODEL)
        :param data_path: path to data (default: utils.paths.DB_LOB)
        :param input_name: name of input tensor (default: None)
        :param output_name: name of output tensor (default: None)
        :param input_shape: input shape (default: None)
        :param allow_gpu_growth: allow GPU growth (default: False)
        """

        assert os.path.exists(model_path), f"{model_path} not found"
        assert not data_path or os.path.exists(data_path), \
            f"{data_path} not found"

        if allow_gpu_growth or ".engine" in model_path:
            options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=options)
            self.sess = tf.Session(config=config)
            self.sess.__enter__()

        if ".h5" in model_path:
            self._keras_init(model_path)
        elif ".pb" in model_path:
            self._tf_init(model_path, input_name, output_name, input_shape)
        elif ".engine" in model_path:
            self._trt_init(model_path, input_name, output_name, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")

        self._db = {}
        self.classifier = None

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

    def _keras_init(self, filepath):
        """Initializes a Keras model
        :param filepath: path to model (.h5)
        """

        self.MODE = "keras"
        self.facenet = tf.keras.models.load_model(filepath)
        self.img_shape = self.facenet.input_shape[1:3]

    def _tf_init(self, filepath, input_name, output_name, input_shape):
        """Initializes a TensorFlow model
        :param filepath: path to model (.pb)
        :param input_name: name of input tensor
        :param output_name: name of output tensor
        :param input_shape: input shape for facenet
        """

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
                input_tensor = self.facenet.get_tensor_by_name(self.input_name)
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

    def update_data(self, person, embeddings, train_knn=True):
        """Updates data property
        :param person: new entry
        :param embeddings: new entry's list of embeddings
        :param train_knn: train K-NN (default: True)
        """
        screen_data(person, embeddings)

        self._db[person] = np.array(embeddings).reshape(len(embeddings), -1)
        self._stripped_names.append(strip_id(person))

        if train_knn:
            self._train_classifier()

    def set_data(self, data, metadata):
        """Sets data property
        :param data: new data in form {name: embedding vector, ...}
        :param metadata: data metadata
        """
        assert metadata, "metadata must be provided"

        self._db = {}
        self._stripped_names = []
        self.data_cfg = metadata

        self.dist_metric = DistMetric(self.data_cfg["metric"],
                                      self.data_cfg["normalize"],
                                      self.data_cfg.get("mean"))
        self.alpha = self.data_cfg["alpha"]
        self.img_norm = self.data_cfg["img_norm"]

        if data:
            for person, embed in data.items():
                self.update_data(person, embed, train_knn=False)
            self._train_classifier()

    @property
    def metadata(self):
        return {"metric": self.dist_metric.metric,
                "normalize": self.dist_metric.normalize,
                "alpha": self.alpha,
                "img_norm": self.img_norm}

    def _train_classifier(self):
        """Trains person classifier"""
        try:
            self.classifier = neighbors.NearestNeighbors(
                radius=self.alpha, metric=self.dist_metric.metric, n_jobs=-1)
            self.classifier.fit(np.squeeze(list(self.data.values()), axis=1))
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
        :returns: embedding, is recognized, best match, distance
        """
        start = timer()

        embed = None
        is_recognized = None
        best_match = None
        dist = None
        face = None

        try:
            embeds, face = self.predict(img, *args, **kwargs)
            dists, idxs = self.classifier.radius_neighbors(embeds)
            dists = dists.flatten()[0]
            idxs = idxs.flatten()[0]

            if len(idxs) != 0:
                matches = np.take(self._stripped_names, idxs).tolist()
                key = lambda i: matches.count(matches[i]) + eps / dists[i]
                best_idx = max(range(len(matches)), key=key)

                best_match, dist = matches[best_idx], dists[best_idx]
                is_recognized = dist <= self.alpha
                info = colored(f"{best_match} - {round(dist, 4)}",
                               color="green" if is_recognized else "red")

            else:
                info = colored("no match found", color="red")

            if verbose:
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
        return embed, is_recognized, best_match, dist, face, elapsed

    def real_time_recognize(self, width=640, height=360, dynamic_log=False,
                            pbar=False, resize=1., detector="mtcnn+haarcascade",
                            data_mutable=False, socket=None, flip=False):
        """Real-time facial recognition
        :param width: width of frame (default: 640)
        :param height: height of frame (default: 360)
        :param dynamic_log: use dynamic database (default: False)
        :param pbar: use progress bar or not (default: False)
        :param resize: resize scale (default: 1. = no resize)
        :param detector: face detector type (default: "mtcnn+haarcascade")
        :param data_mutable: update database iff requested (default: False)
        :param socket: socket address (dev only)
        :param flip: whether to flip horizontally or not (default: False)
        """

        assert self._db, "data must be provided"
        assert 0. <= resize <= 1., "resize must be in [0., 1.]"

        logger = Logger()
        websocket = Websocket(socket) if socket else None
        ipbar = IntegratedLCDProgressBar(logger, websocket) if pbar else None
        ilogger = IntegratedLogger(self, logger, ipbar, websocket,
                                   data_mutable, dynamic_log)

        graphics_controller = GraphicsRenderer(width, height, resize)
        cap = Camera(width, height)

        detector = FaceDetector(detector, self.img_shape, min_face_size=240)

        while True:
            _, frame = cap.read()
            original_frame = frame.copy()

            # resize frame
            if resize != 1:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            # facial detection and recognition
            result = self.recognize(frame, detector, flip=flip)
            embed, is_recognized, best_match, dist, face, elapsed = result

            # graphics, logging, lcd, etc.
            ilogger.log_activity(best_match, embed, dist)
            graphics_controller.add_graphics(original_frame, face,
                                             is_recognized, best_match,
                                             elapsed)

            cv2.imshow("AI Security v2021.0.1", original_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        ilogger.close()

        return ilogger
