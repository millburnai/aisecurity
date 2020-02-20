"""

"aisecurity.facenet"

Facial recognition with FaceNet in Keras or TF-TRT.

Reference paper: https://arxiv.org/pdf/1503.03832.pdf

"""

import asyncio
import itertools
import json
import os
from timeit import default_timer as timer
import warnings

import cv2
import keras
from keras import backend as K
import numpy as np
from sklearn import neighbors
import tensorflow as tf
from termcolor import cprint

from aisecurity.dataflow.data import retrieve_embeds, dump_and_encrypt
from aisecurity.db import log
from aisecurity.optim import engine
from aisecurity.hardware import keypad, lcd
from aisecurity.utils.distance import DistMetric
from aisecurity.utils.events import print_time, run_async_method
from aisecurity.utils.paths import DATABASE, DATABASE_INFO, DEFAULT_MODEL, CONFIG_HOME
from aisecurity.utils.visuals import get_video_cap, add_graphics
from aisecurity.face.detection import detector_init
from aisecurity.face.preprocessing import IMG_CONSTANTS, normalize, crop_face


################################ FaceNet ###############################
class FaceNet:
    """Class implementation of FaceNet"""


    # HYPERPARAMETERS
    HYPERPARAMS = {
        "alpha": 0.75,
        "mtcnn_alpha": 0.9
    }

    # PRE-BUILT MODEL CONFIGS
    MODELS = json.load(open(CONFIG_HOME + "/config/models.json"))


    # INITS
    @print_time("Model load time")
    def __init__(self, model_path=DEFAULT_MODEL, data_path=DATABASE, sess=None, input_name=None, output_name=None,
                 input_shape=None, **hyperparams):
        """Initializes FaceNet object

        :param model_path: path to model (default: aisecurity.utils.paths.DEFAULT_MODEL)
        :param data_path: path to data(default: aisecurity.utils.paths.DATABASE)
        :param sess: tf.Session to use (default: None)
        :param input_name: name of input tensor-- only required if using (TF)TRT non-default model (default: None)
        :param output_name: name of output tensor-- only required if using (TF)TRT non-default model (default: None)
        :param input_shape: input shape-- only required if using (TF)TRT non-default model (default: None)
        :param hyperparams: hyperparameters to override FaceNet.HYPERPARAMS

        """

        assert os.path.exists(model_path), "{} not found".format(model_path)
        assert not data_path or os.path.exists(data_path), "{} not found".format(data_path)

        if ".pb" in model_path:
            self._tf_trt_init(model_path, input_name, output_name, sess, input_shape)
        elif ".h5" in model_path:
            self._keras_init(model_path)
        elif ".engine" in model_path:
            self._trt_init(model_path, input_name, output_name, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")

        self._db = {}
        self._knn = None

        if data_path:
            self.set_data(retrieve_embeds(data_path), config=DATABASE_INFO)
        else:
            warnings.warn("data not set. Set it manually with set_data to use FaceNet")
        self.set_dist_metric("auto")

        self.HYPERPARAMS.update(hyperparams)

    # KERAS INIT
    def _keras_init(self, filepath):
        """Initializes a Keras model

        :param filepath: path to model (.h5)

        """

        self.MODE = "keras"

        self.facenet = keras.models.load_model(filepath)

        self.img_norm = self.MODELS["_default"]["params"]["img_norm"]
        IMG_CONSTANTS["img_size"] = self.facenet.input_shape[1:3]


    # TF-TRT INIT
    def _tf_trt_init(self, filepath, input_name, output_name, sess, input_shape):
        """Initializes a TF-TRT model

        :param filepath: path to model (.pb)
        :param input_name: name of input tensor
        :param output_name: name of output tensor
        :param sess: tf.Session to enter
        :param input_shape: input shape for facenet

        """

        self.MODE = "tf-trt"

        graph_def = self.get_frozen_graph(filepath)

        if sess:
            self.sess = sess
            K.set_session(self.sess)
        else:
            self.sess = K.get_session()

        tf.import_graph_def(graph_def, name="")

        self._tensor_init(model_name=filepath, input_name=input_name, output_name=output_name)

        self.facenet = self.sess.graph

        try:
            if input_shape is not None:
                IMG_CONSTANTS["img_size"] = input_shape
            else:
                IMG_CONSTANTS["img_size"] = tuple(
                    self.facenet.get_tensor_by_name(self.input_name).get_shape().as_list()[1:3]
                )
        except ValueError:
            warnings.warn("Input tensor size not detected. Default size is {}".format(IMG_CONSTANTS["img_size"]))

    def _tensor_init(self, model_name, input_name, output_name):
        """Initializes tensors (TF-TRT or TRT modes only)

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
        self.params = self.model_config["params"]
        self.img_norm = self.params["img_norm"]

        if not self.input_name:
            self.input_name = input_name
        elif not self.output_name:
            self.output_name = output_name

        assert self.input_name and self.output_name, "I/O tensors for {} not detected or provided".format(model_name)

    # TRT INIT
    def _trt_init(self, filepath, input_name, output_name, input_shape):
        """TensorRT initialization

        :param filepath: path to serialized engine (not portable across GPUs or platforms)
        :param input_name: name of input to network
        :param output_name: name of output to network
        :param input_shape: input shape (channels first)

        """

        assert engine.INIT_SUCCESS, "tensorrt or pycuda import failed: trt mode not available"

        self.MODE = "trt"

        tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))).__enter__()
        # needed to fix https://stackoverflow.com/questions/58756919/python-tensorrt-cudnn-status-mapping-error-error

        self.facenet = engine.CudaEngine(filepath, input_name, output_name, input_shape)

        self.img_norm = self.MODELS["_default"]["params"]["img_norm"]
        IMG_CONSTANTS["img_size"] = tuple(reversed(self.facenet.input_shape))[:-1]


    # MUTATORS
    @staticmethod
    def _screen_data(key, value):
        """Checks if key-value pair is valid for data dict

        :param key: new key
        :param value: new value

        """

        assert isinstance(key, str), "data keys must be person names"

        for embed in value:
            embed = np.asarray(embed)
            is_vector = embed.ndim <= 2 and (1 in embed.shape or embed.ndim == 1)
            assert is_vector, "each value must be a vectorized embedding, got shape {}".format(embed.shape)

        return key, value

    def update_data(self, person, embeddings, train_knn=True):
        """Updates data property

        :param person: new entry
        :param embeddings: new entry's list of embeddings
        :param train_knn: whether or not to train K-NN (default: True)

        """

        person, embeddings = self._screen_data(person, embeddings)
        embeddings = [np.array(embed).reshape(-1, ) for embed in embeddings]

        if person in self.data:
            self._db[person].extend(embeddings)
        else:
            self._db[person] = embeddings

        if train_knn:
            self._train_knn()

    def set_data(self, data, config=None):
        """Sets data property

        :param data: new data in form {name: embedding vector, ...}
        :param config: data config dict with the entry "metric": <DistMetric str constructor> (default: None)

        """

        assert data, "data must be provided"

        for person, embed in data.items():
            self.update_data(person, embed, train_knn=False)
        self._train_knn()

        if config is None:
            warnings.warn("data config missing. Distance metric not detected")
        else:
            self.data_config = config
            self.cfg_dist_metric = self.data_config["metric"]

    def set_dist_metric(self, dist_metric):
        """Sets distance metric for FaceNet

        :param dist_metric: DistMetric object or str constructor, or "auto+{whatever}" to detect from self.data_config

        """

        # set distance metric
        if isinstance(dist_metric, DistMetric):
            self.dist_metric = dist_metric

        elif isinstance(dist_metric, str):
            if "auto" in dist_metric:
                constructor = self.cfg_dist_metric
                if "+" in dist_metric:
                    constructor += dist_metric[dist_metric.find("+"):]
            else:
                constructor = dist_metric

            self.dist_metric = DistMetric(constructor, data=list(self.data.values()), axis=0)

        else:
            raise ValueError("{} not a supported dist metric".format(dist_metric))

        # check against data config
        self.ignore = {0: self.dist_metric.get_config()}

        if self.cfg_dist_metric != self.dist_metric.get_config():
            self.ignore[1] = self.cfg_dist_metric

            warnings.warn("provided DistMetric ({}) is not the same as the data config metric ({}) ".format(
                self.dist_metric.get_config(), self.cfg_dist_metric))

    def _train_knn(self):
        """Trains K-Nearest-Neighbors"""
        try:
            self.expanded_names = []
            self.expanded_embeds = []

            for name, embeddings in self.data.items():
                for idx, embedding in enumerate(embeddings):
                    self.expanded_names.append(name)
                    self.expanded_embeds.append(embedding)

            # always use minkowski distance, other metrics are just normalizing before minkowski to act
            # as the desired metric (ex: cosine)
            n_neighbors = len(self.expanded_names) // len(set(self.expanded_names))
            self._knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            self._knn.fit(self.expanded_embeds, self.expanded_names)

        except (AttributeError, ValueError):
            raise ValueError("Current model incompatible with database")


    # RETRIEVERS
    @property
    def data(self):
        """Property for static database

        :returns: self._db

        """

        return self._db

    @staticmethod
    def get_frozen_graph(path):
        """Gets frozen graph from .pb file (TF-TRT only)

        :param path: path to .pb frozen graph file
        :returns: tf.GraphDef object

        """

        with tf.gfile.FastGFile(path, "rb") as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())
        return graph_def

    def _make_feed_dict(self, img):
        """Makes feed dict for sess.run (TF-TRT only)

        :param img: image input
        :returns: feed dict

        """

        feed_dict = {self.input_name: np.expand_dims(img, axis=0)}
        for tensor, value in self.params["feed_dict"].items():
            feed_dict[tensor] = value
        return feed_dict

    @print_time("Embedding time")
    def embed(self, img):
        """Embeds cropped face

        :param img: img as a cropped face with shape (h, w, 3)

        """

        if self.MODE == "keras":
            return self.facenet.predict(np.expand_dims(img, axis=0)).reshape(1, -1)
        elif self.MODE == "tf-trt":
            output_tensor = self.facenet.get_tensor_by_name(self.output_name)
            return self.sess.run(output_tensor, feed_dict=self._make_feed_dict(img)).reshape(1, -1)
        elif self.MODE == "trt":
            return self.facenet.inference(np.expand_dims(img, axis=0), output_shape=(1, -1))

    def predict(self, path_or_img, face_detector="mtcnn", margin=IMG_CONSTANTS["margin"]):
        """Embeds and normalizes an image from path or array

        :param path_or_img: path or image to predict on
        :param face_detector: face detector (either mtcnn, haarcascade, or None) (default: "mtcnn")
        :param margin: margin for MTCNN face cropping (default: aisecurity.preprocessing.IMG_CONSTANTS["margin"])
        :returns: normalized embeddings, facial coordinates

        """

        cropped_face, face_coords = crop_face(path_or_img, margin, face_detector, alpha=self.HYPERPARAMS["mtcnn_alpha"])
        if face_coords == -1:
            return itertools.repeat(-1, 2)  # exit code: failure to detect face

        normalized_face = normalize(cropped_face, mode=self.img_norm)

        raw_embedding = self.embed(normalized_face)
        normalized_embedding = self.dist_metric(raw_embedding)

        return normalized_embedding, face_coords


    # FACIAL RECOGNITION HELPER
    def recognize(self, img, **kwargs):
        """Facial recognition

        :param img: image array
        :param kwargs: named arguments to self.get_embeds (will be passed to self.predict)
        :returns: embedding, is recognized (bool), best match from database(s), distance

        """

        exit_failure = itertools.repeat(-1, 6)

        try:
            assert self._db, "data must be provided"

            start = timer()

            embedding, face = self.predict(img, **kwargs)
            if face == -1:  # no face detected
                return exit_failure

            best_match = self._knn.predict(embedding)[0]
            best_embed = self.expanded_embeds[self.expanded_names.index(best_match)]

            dist = self.dist_metric(embedding, best_embed, mode="calc+norm", ignore=self.ignore)
            is_recognized = dist <= FaceNet.HYPERPARAMS["alpha"]

            elapsed = round(timer() - start, 4)

            return embedding, is_recognized, best_match, dist, face, elapsed

        except (ValueError, cv2.error) as error:  # error-handling using names is unstable-- change later
            if "query data dimension" in str(error):
                raise ValueError("Current model incompatible with database")
            elif "empty" in str(error):
                print("Image refresh rate too high")
                return exit_failure
            elif "opencv" in str(error):
                print("Failed to capture frame")
                return exit_failure
            else:
                raise error


    # REAL-TIME FACIAL RECOGNITION HELPER
    async def _real_time_recognize(self, width, height, dist_metric, logging, use_dynamic, use_picam, use_graphics,
                                   use_lcd, use_keypad, framerate, resize, flip, device, face_detector, data_mutability,
                                   socket):
        """Real-time facial recognition under the hood (dev use only)

        :param width: width of frame (only matters if use_graphics is True)
        :param height: height of frame (only matters if use_graphics is True)
        :param metric: DistMetric object
        :param logging: logging type-- None, "firebase", or "mysql"
        :param use_dynamic: use dynamic database for visitors or not
        :param use_picam: use picamera or not
        :param use_graphics: display video feed or not
        :param use_lcd: use LCD or not. If LCD is not connected, will default to LCD simulation and warn
        :param use_keypad: use keypad or not. If keypad not connected, will default to False and warn
        :param framerate: frame rate (recommended <120)
        :param resize: resize scale (float between 0. and 1.)
        :param flip: flip method: +1 = +90º rotation
        :param device: camera device (/dev/video{device})
        :param face_detector: face detector type ("mtcnn" or "haarcascade")
        :param data_mutability: level on which static data is mutable (0, 1, or 2)
        :param socket: in dev
        :returns: number of frames elapsed

        """

        # INITS
        log.init(logging, flush=True)
        if logging:
            log.server_init()
        if use_lcd:
            lcd.init()
        if use_keypad:
            keypad.init()
        if dist_metric:
            self.set_dist_metric(dist_metric)
        if resize:
            face_width, face_height = width * resize, height * resize
        else:
            face_width, face_height = width, height

        cap = get_video_cap(width, height, picamera=use_picam, framerate=framerate, flip=flip, device=device)
        assert cap.isOpened(), "video capture failed to initialize"

        detector_init(min_face_size=int(0.5 * (face_width + face_height) / 2))
        # face needs to fill at least ~1/2 of the frame

        missed_frames = 0
        frames = 0

        last_gpu_checkup = timer()

        # CAM LOOP
        while True:
            _, frame = cap.read()
            original_frame = frame.copy()

            if socket and frames - missed_frames % 100 == 0:
                socket.send(str(frames - missed_frames))

            if resize:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            if use_picam:
                last_gpu_checkup = self.keep_gpu_warm(frame, frames, last_gpu_checkup, use_lcd)

            # facial detection and recognition
            embedding, is_recognized, best_match, dist, face, elapsed = self.recognize(frame, face_detector=face_detector)

            if face != -1:
                print("%s: %.4f (%s)%s" % (self.dist_metric, dist, best_match, "" if is_recognized else " !"))

                # add graphics, lcd, and log
                if use_graphics:
                    add_graphics(original_frame, face, width, height, is_recognized, best_match, resize, elapsed)

                if use_lcd and is_recognized:
                    lcd.PROGRESS_BAR.update(previous_msg="Recognizing...")


                if frames > 5:  # five frames before logging starts
                    self.log_activity(logging, is_recognized, best_match, embedding, use_dynamic, data_mutability)
                    log.DISTS.append(dist)

            else:
                missed_frames += 1
                if missed_frames > log.THRESHOLDS["missed_frames"]:
                    missed_frames = 0
                    log.flush_current(mode="known+unknown", flush_times=False)
                print("No face detected")

            if use_graphics:
                cv2.imshow("AI Security v0.9a", original_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frames += 1
            await asyncio.sleep(1e-6)

        cap.release()
        cv2.destroyAllWindows()

        if data_mutability == 2:
            dump_and_encrypt(self.data, DATABASE, encrypt=DATABASE_INFO["encrypted"])

        return frames


    # REAL-TIME FACIAL RECOGNITION
    def real_time_recognize(self, width=640, height=360, dist_metric="euclidean+l2_normalize", logging=None,
                            use_dynamic=False, use_picam=False, use_graphics=True, use_lcd=False, use_keypad=False,
                            framerate=20, resize=None, flip=0, device=0, face_detector="mtcnn", data_mutability=0,
                            socket=None):
        """Real-time facial recognition

        :param width: width of frame (only matters if use_graphics is True) (default: 640)
        :param height: height of frame (only matters if use_graphics is True) (default: 360)
        :param dist_metric: DistMetric object or str distance metric (default: "euclidean+l2_normalize")
        :param logging: logging type-- None, "firebase", or "mysql" (default: None)
        :param use_dynamic: use dynamic database for visitors or not (default: False)
        :param use_picam: use picamera or not (default: False)
        :param use_graphics: display video feed or not (default: True)
        :param use_lcd: use LCD or not. If LCD isn't connected, will default to LCD simulation and warn (default: False)
        :param use_keypad: use keypad or not. If keypad not connected, will default to False and warn (default: False)
        :param framerate: frame rate, only matters if use_picamera is True (recommended <120) (default: 20)
        :param resize: resize scale (float between 0. and 1.) (default: None)
        :param flip: flip method: +1 = +90º rotation (default: 0)
        :param device: camera device (/dev/video{device}) (default: 0)
        :param face_detector: face detector type ("mtcnn" or "haarcascade") (default: "mtcnn")
        :param data_mutability: level of data mutability (0-2) (default: 0)
                                0: No data mutability
                                1: Prompt for verification on recognition and update database
                                2: Write updated data at end of real_time_recognize
        :param socket: in dev

        """

        assert width > 0 and height > 0, "width and height must be positive integers"
        assert not logging or logging == "mysql" or logging == "firebase", "only mysql and firebase logging supported"
        assert 0 < framerate <= 120, "framerate must be between 0 and 120"
        assert resize is None or 0. < resize < 1., "resize must be between 0 and 1"
        assert face_detector in ("mtcnn", "haarcascade"), "only mtcnn and haarcascade supported"

        run_async_method(
            self._real_time_recognize,
            width=width, height=height, dist_metric=dist_metric, logging=logging, use_dynamic=use_dynamic,
            use_picam=use_picam, use_graphics=use_graphics, use_lcd=use_lcd, use_keypad=use_keypad, framerate=framerate,
            resize=resize, flip=flip, device=device, face_detector=face_detector, data_mutability=data_mutability,
            socket=socket,
        )


    # LOGGING
    def log_activity(self, logging, is_recognized, best_match, embedding, use_dynamic, data_mutability):
        """Logs facial recognition activity

        :param logging: logging type-- None, "firebase", or "mysql"
        :param is_recognized: whether face was recognized or not
        :param best_match: best match from database
        :param mode: logging type: "firebase" or "mysql"
        :param embedding: embedding vector
        :param use_dynamic: use dynamic database or not
        :param data_mutability: level of static data mutability

        """

        log.update_current_logs(is_recognized, best_match)

        update_recognized = log.NUM_RECOGNIZED >= log.THRESHOLDS["num_recognized"] and log.cooldown_ok(log.LAST_LOGGED)
        update_unrecognized = log.NUM_UNKNOWN >= log.THRESHOLDS["num_unknown"] and log.cooldown_ok(log.UNK_LAST_LOGGED)

        if update_recognized:
            if log.get_percent_diff(best_match, log.CURRENT_LOG) <= log.THRESHOLDS["percent_diff"]:
                recognized_person = log.get_mode(log.CURRENT_LOG)
                log.log_person(logging, recognized_person, times=log.CURRENT_LOG[recognized_person])

                lcd.add_lcd_display(best_match, log.USE_SERVER)  # will silently fail if lcd not supported

        elif update_unrecognized:
            log.log_unknown(logging, "<DEPRECATED>")

            if use_dynamic:
                visitor_num = len([person for person in self._db if "visitor" in person]) + 1
                self.update_data("visitor_{}".format(visitor_num), [embedding])

                cprint("Visitor {} activity logged".format(visitor_num), color="magenta", attrs=["bold"])

        if data_mutability and (update_recognized or update_unrecognized):
            is_correct = input("Are you {}? ".format(best_match.replace("_", " ").title())).lower()
            # TODO: replace with input from keyboard

            if len(is_correct) == 0 or is_correct[0] == "y":
                self.update_data(best_match, [embedding])
            else:
                name = input("Who are you? ").lower().replace(" ", "_")

                if name in self.data:
                    self.update_data(name, [embedding])
                    cprint("Static entry for '{}' updated".format(name), color="blue", attrs=["bold"])
                else:
                    cprint("'{}' is not in database".format(name), attrs=["bold"])


    # COMPUTATION CHECK
    def keep_gpu_warm(self, frame, frames, last_gpu_checkup, use_lcd):
        """Keeps GPU computations running so that facial recognition speed stays constant. Only needed on Jetson Nano

        :param frame: frame as array
        :param frames: number of frames elapsed
        :param last_gpu_checkup: last computation check time
        :param use_lcd: use LCD or not
        :returns: this computation check time

        """

        next_check = log.THRESHOLDS["missed_frames"]

        if frames == 0 or timer() - last_gpu_checkup > next_check:
            self.embed(normalize(cv2.resize(frame, IMG_CONSTANTS["img_size"]), mode=self.img_norm))
            print("Regular computation check")

            last_gpu_checkup = timer()

            if use_lcd:
                lcd.LCD_DEVICE.clear()

        elif not (timer() - log.LAST_LOGGED > next_check or timer() - log.UNK_LAST_LOGGED > next_check):
            last_gpu_checkup = timer()

        return last_gpu_checkup
