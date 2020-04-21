"""

"aisecurity.facenet"

Facial recognition with FaceNet in Keras, TensorFlow, or TensorRT.

Reference paper: https://arxiv.org/pdf/1503.03832.pdf

"""

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

from aisecurity.dataflow.loader import print_time, retrieve_embeds
from aisecurity.db import log, connection
from aisecurity.optim import engine
from aisecurity.utils import lcd
from aisecurity.utils.distance import DistMetric
from aisecurity.utils.paths import DATABASE, DATABASE_INFO, DEFAULT_MODEL, CONFIG_HOME
from aisecurity.utils.visuals import get_video_cap, add_graphics
from aisecurity.face.detection import detector_init
from aisecurity.face.preprocessing import set_img_shape, normalize, crop_face, IMG_SHAPE


################################ FaceNet ###############################
class FaceNet:
    """Class implementation of FaceNet"""

    # HYPERPARAMETERS
    ALPHA = 0.75


    # PRE-BUILT MODEL CONFIGS
    MODELS = json.load(open(CONFIG_HOME + "/config/models.json", encoding="utf-8"))


    # INITS
    @print_time("Model load time")
    def __init__(self, model_path=DEFAULT_MODEL, data_path=DATABASE, sess=None, input_name=None, output_name=None,
                 input_shape=None):
        """Initializes FaceNet object
        :param model_path: path to model (default: aisecurity.utils.paths.DEFAULT_MODEL)
        :param data_path: path to data(default: aisecurity.utils.paths.DATABASE)
        :param sess: tf.Session to use (default: None)
        :param input_name: name of input tensor-- only required if using TF/TRT non-default model (default: None)
        :param output_name: name of output tensor-- only required if using TF/TRT non-default model (default: None)
        :param input_shape: input shape-- only required if using TF/TRT non-default model (default: None)
        """

        assert os.path.exists(model_path), "{} not found".format(model_path)
        assert not data_path or os.path.exists(data_path), "{} not found".format(data_path)

        if ".h5" in model_path:
            self._keras_init(model_path)
        elif ".pb" in model_path:
            self._tf_init(model_path, input_name, output_name, sess, input_shape)
        elif ".engine" in model_path:
            self._trt_init(model_path, input_name, output_name, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")

        self.img_norm = self.MODELS["_default"]["img_norm"]
        for model in self.MODELS:
            if model_path[model_path.rfind("/")+1:model_path.rfind(".")] in model:
                self.img_norm = self.MODELS[model]["img_norm"]

        self._db = {}
        self._knn = None

        if data_path:
            self.set_data(retrieve_embeds(data_path), config=DATABASE_INFO)
        else:
            warnings.warn("data not set. Set it manually with set_data to use FaceNet")
        self.set_dist_metric("auto")


    # KERAS INIT
    def _keras_init(self, filepath):
        """Initializes a Keras model
        :param filepath: path to model (.h5)
        """

        self.MODE = "keras"

        self.facenet = keras.models.load_model(filepath)

        set_img_shape(self.facenet.input_shape[1:3])


    # TENSORFLOW INIT
    def _tf_init(self, filepath, input_name, output_name, sess, input_shape):
        """Initializes a TensorFlow model
        :param filepath: path to model (.pb)
        :param input_name: name of input tensor
        :param output_name: name of output tensor
        :param sess: tf.Session to enter
        :param input_shape: input shape for facenet
        """

        self.MODE = "tf"

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
                assert input_shape[-1] == 3, "input shape must be channels-last for tensorflow mode"
                set_img_shape(input_shape[:-1])
            else:
                set_img_shape(self.facenet.get_tensor_by_name(self.input_name).get_shape().as_list()[1:3])

        except ValueError:
            warnings.warn("Input tensor size not detected. Default size is {}".format(IMG_SHAPE))

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

        set_img_shape(list(reversed(self.facenet.input_shape))[:-1])


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
            is_vector = np.prod(embed.shape) == embed.flatten().shape
            assert is_vector, "each value must be a vectorized embedding, got shape {}".format(embed.shape)

        return key, value

    def update_data(self, person, embeddings, train_knn=True):
        """Updates data property
        :param person: new entry
        :param embeddings: new entry's list of embeddings
        :param train_knn: whether or not to train K-NN (default: True)
        """

        person, embeddings = self._screen_data(person, embeddings)
        embeddings = np.array(embeddings).reshape(len(embeddings), -1).tolist()

        if not self.data:
            self._db = {}

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

        self._db = None

        if data:
            for person, embed in data.items():
                self.update_data(person, embed, train_knn=False)
            self._train_knn()

            if config is None:
                warnings.warn("data config missing. Distance metric not detected")
            else:
                self.data_cfg = config

    def set_dist_metric(self, dist_metric):
        """Sets distance metric for FaceNet
        :param dist_metric: DistMetric object or str constructor, or "auto+{whatever}" to detect from self.data_cfg
        """

        cfg_metric = self.data_cfg["metric"]

        # set distance metric
        if isinstance(dist_metric, DistMetric):
            self.dist_metric = dist_metric

        elif isinstance(dist_metric, str):
            if "auto" in dist_metric:
                constructor = cfg_metric
                if "+" in dist_metric:
                    constructor += dist_metric[dist_metric.find("+"):]
            else:
                constructor = dist_metric

            self.dist_metric = DistMetric(constructor, data=list(self.data.values()), axis=0)

        else:
            raise ValueError("{} not a supported dist metric".format(dist_metric))

        # check against data config
        self.ignore_norms = [self.dist_metric.get_config()]

        if cfg_metric != self.dist_metric.get_config():
            self.ignore_norms.append(self.data_cfg)

            warnings.warn("provided DistMetric ({}) is not the same as the data config metric ({}) ".format(
                self.dist_metric.get_config(), cfg_metric))

    def _train_knn(self):
        """Trains K-Nearest-Neighbors"""
        try:
            self.expanded_names = []
            self.expanded_embeds = []

            for name, embeddings in self.data.items():
                for embed in embeddings:
                    self.expanded_names.append(name)
                    self.expanded_embeds.append(embed)

            n_neighbors = len(self.expanded_names) // len(set(self.expanded_names))
            # auto-detect number of neighbors as minimum number of embeddings per person
            self._knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            # always use minkowski distance, other metrics are just normalizing to act as the desired metric
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
        """Gets frozen graph from .pb file (TF only)
        :param path: path to .pb frozen graph file
        :returns: tf.GraphDef object
        """

        with tf.gfile.FastGFile(path, "rb") as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())
        return graph_def

    def embed(self, imgs):
        """Embeds cropped face
        :param imgs: list of cropped faces with shape (batch_size, h, w, 3)
        :returns: embedding as array with shape (1, -1)
        """

        if self.MODE == "keras":
            embeds = self.facenet.predict(imgs, batch_size=len(imgs))
        elif self.MODE == "tf":
            output_tensor = self.facenet.get_tensor_by_name(self.output_name)
            embeds = self.sess.run(output_tensor, feed_dict={self.input_name: imgs})
        elif self.MODE == "trt":
            embeds = self.facenet.inference(imgs)

        return embeds.reshape(len(imgs), -1)

    def predict(self, img, detector="both", margin=10, rotations=None):
        """Embeds and normalizes an image from path or array
        :param img: image to be predicted on (BGR image)
        :param detector: face detector (either mtcnn, haarcascade, or None) (default: "both")
        :param margin: margin for MTCNN face cropping (default: 10)
        :param rotations: array of rotations to be applied to face (default: None)
        :returns: normalized embeddings, facial coordinates
        """

        cropped_faces, face_coords = crop_face(img[:, :, ::-1], margin, detector, rotations=rotations)
        start = timer()

        assert cropped_faces.shape[1:] == (*IMG_SHAPE, 3), "no face detected"

        raw_embeddings = np.expand_dims(self.embed(normalize(cropped_faces, mode=self.img_norm)), axis=1)
        normalized_embeddings = self.dist_metric.apply_norms(*raw_embeddings)

        message = "{} vector{}".format(len(normalized_embeddings), "s" if len(normalized_embeddings) > 1 else "")
        print("Embedding time ({}): \033[1m{} ms\033[0m".format(message, round(1000. * (timer() - start), 2)))

        return normalized_embeddings, face_coords


    # FACIAL RECOGNITION HELPER
    def recognize(self, img, **kwargs):
        """Facial recognition
        :param img: image array in BGR mode
        :param kwargs: named arguments to self.get_embeds (will be passed to self.predict)
        :returns: embedding, is recognized (bool), best match from database(s), distance
        """

        def analyze_embeds(embeds):
            analysis = {"best_match": [], "dists": [], "is_recognized": []}
            for embed in embeds:
                analysis["best_match"].append(self._knn.predict(embed)[0])
                best_embed = self.expanded_embeds[self.expanded_names.index(analysis["best_match"][-1])]

                analysis["dists"].append(self.dist_metric.distance(embed, best_embed, ignore_norms=self.ignore_norms))
                analysis["is_recognized"].append(analysis["dists"][-1] <= FaceNet.ALPHA)

            return analysis

        start = timer()
        embed, is_recognized, best_match, dist, face, elapsed = None, None, None, None, None, None

        try:
            embeds, face = self.predict(img, **kwargs)
            analysis = analyze_embeds(embeds)

            if len(embeds) > 1:
                best_match = max(analysis["best_match"], key=analysis["best_match"].count)

                best_match_idxs = [idx for idx, person in enumerate(analysis["best_match"]) if person == best_match]
                min_index = min(best_match_idxs, key=lambda idx: analysis["dists"][idx])
                # index associated with minimum distance best_match embedding

            else:
                best_match = analysis["best_match"][0]
                min_index = 0

            embed = embeds[min_index]
            dist = analysis["dists"][min_index]
            is_recognized = analysis["is_recognized"][min_index]

            print("%s: \033[1m%.4f (%s)%s\033[0m" % (self.dist_metric, dist, best_match, "" if is_recognized else " !"))

        except (ValueError, AssertionError, cv2.error) as error:
            if "query data dimension" in str(error):
                raise ValueError("Current model incompatible with database")
            elif isinstance(error, cv2.error) and "resize" in str(error):
                print("Frame capture failed")
            elif not isinstance(error, AssertionError):
                raise error

        elapsed = round(1000. * (timer() - start), 4)
        return embed, is_recognized, best_match, dist, face, elapsed


    # REAL-TIME FACIAL RECOGNITION
    def real_time_recognize(self, width=640, height=360, dist_metric=None, logging=None, dynamic_log=False, pbar=False,
                            resize=None, flip=0, detector="both", data_mutable=False, socket=None, rotations=None,
                            device=0):
        """Real-time facial recognition
        :param width: width of frame (only matters if use_graphics is True) (default: 640)
        :param height: height of frame (only matters if use_graphics is True) (default: 360)
        :param dist_metric: DistMetric object or str distance metric (default: this.dist_metric)
        :param logging: logging type-- None, "firebase", or "mysql" (default: None)
        :param dynamic_log: use dynamic database for visitors or not (default: False)
        :param pbar: use progress bar or not. If Pi isn't reachable, will default to LCD simulation (default: False)
        :param resize: resize scale (float between 0. and 1.) (default: None)
        :param flip: flip method: +1 = +90º rotation (default: 0)
        :param detector: face detector type ("mtcnn", "haarcascade", "both") (default: "both")
        :param data_mutable: if true, prompt for verification on recognition and update database (default: False)
        :param socket: socket address (dev only)
        :param rotations: rotations to be applied to face (-1 is horizontal flip) (default: None)
        :param device: video file to read from (passing an int will use /dev/video{device}) (default: 0)
        """

        # INITS
        assert self._db, "data must be provided"
        log.init(logging, flush=True)
        if dist_metric:
            self.set_dist_metric(dist_metric)
        if socket:
            connection.init(socket)
        if pbar:
            lcd.init()
        if resize:
            assert 0. <= resize <= 1., "resize must be in [0., 1.]"
            face_width, face_height = width * resize, height * resize
        else:
            face_width, face_height = width, height

        cap = get_video_cap(width, height, flip, device)
        detector_init(min_face_size=0.5 * (face_width + face_height) / 2)
        # face needs to fill at least ~1/2 of the frame

        absent_frames = 0
        frames = 0

        # CAM LOOP
        while True:
            _, frame = cap.read()
            original_frame = frame.copy()

            if resize:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            # facial detection and recognition
            embed, is_recognized, best_match, dist, face, elapsed = self.recognize(
                frame, detector=detector, rotations=rotations
            )

            # graphics, logging, lcd, etc.
            absent_frames += self.log_activity(best_match, embed, dynamic_log, data_mutable, pbar, dist, absent_frames)
            add_graphics(original_frame, face, width, height, is_recognized, best_match, resize, elapsed)

            cv2.imshow("AI Security v0.9a", original_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frames += 1

        cap.release()
        cv2.destroyAllWindows()

        return frames


    # LOGGING
    def log_activity(self, best_match, embedding, dynamic_log, data_mutable, pbar, dist, absent_frames):
        """Logs facial recognition activity
        :param best_match: best match from database
        :param mode: logging type: "firebase" or "mysql"
        :param embedding: embedding vector
        :param dynamic_log: use dynamic database or not
        :param data_mutable: static data mutability or not
        :param pbar: use pbar or not
        :param dist: distance between best match and current frame
        """

        if best_match is None:
            absent_frames += 1
            if absent_frames > log.THRESHOLDS["missed_frames"]:
                absent_frames = 0
                log.flush_current(mode="known+unknown", flush_times=False)
            return absent_frames

        is_recognized = dist <= FaceNet.ALPHA
        update_progress, update_recognized, update_unrecognized = log.update(is_recognized, best_match)

        if pbar and update_progress:
            lcd.update_progress(update_recognized)

        if update_recognized and connection.SOCKET:
            connection.send(best_match=best_match)
            connection.receive()

        elif update_unrecognized:
            if pbar:
                lcd.PROGRESS_BAR.reset(message="Recognizing...")

            if dynamic_log:
                visitor_num = len([person for person in self._db if "visitor" in person]) + 1
                self.update_data("visitor_{}".format(visitor_num), [embedding])

                lcd.PROGRESS_BAR.update(amt=np.inf, message="Visitor {} created".format(visitor_num))
                cprint("Visitor {} activity logged".format(visitor_num), color="magenta", attrs=["bold"])

        if data_mutable and (update_recognized or update_unrecognized):
            if connection.SOCKET:
                is_correct = not bool(connection.RECV)
            else:
                user_input = input("Are you {}? ".format(best_match.replace("_", " ").title())).lower()
                is_correct = bool(len(user_input) == 0 or user_input[0] == "y")

            if is_correct:
                self.update_data(best_match, [embedding])
            else:
                if connection.SOCKET:
                    name, connection.RECV = connection.RECV, None
                else:
                    name = input("Who are you? ").lower().replace(" ", "_")

                if name in self.data:
                    self.update_data(name, [embedding])
                    cprint("Static entry for '{}' updated".format(name), color="blue", attrs=["bold"])
                else:
                    cprint("'{}' is not in database".format(name), attrs=["bold"])

        if pbar:
            lcd.check_clear()

        log.DISTS.append(dist)
        return absent_frames
