"""

"aisecurity.facenet"

Facial recognition with FaceNet in Keras or TF-TRT.

Reference paper: https://arxiv.org/pdf/1503.03832.pdf

"""

import asyncio
import os
import time
import warnings

import cv2
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
from sklearn import neighbors
import tensorflow as tf
from termcolor import cprint

from aisecurity.database import log
from aisecurity.privacy.encryptions import DataEncryption
from aisecurity.hardware import keypad, lcd
from aisecurity.utils.distance import DistMetric
from aisecurity.utils.events import timer, HidePrints, run_async_method
from aisecurity.utils.paths import CONFIG_HOME
from aisecurity.data.preprocessing import IMG_CONSTANTS, normalize, crop_faces


# ---------------- FACENET ----------------

class FaceNet:
    """Class implementation of FaceNet"""


    # HYPERPARAMETERS
    HYPERPARAMS = {
        "alpha": 0.75,
        "mtcnn_alpha": 0.9
    }

    # TENSOR CONSTANTS (FOR TF-TRT)
    TENSORS = {
        "ms_celeb_1m": {
            "input": "input_1:0",
            "output": "Bottleneck_BatchNorm/batchnorm/add_1:0"
        },
        "vgg_face_2": {
            "input": "base_input:0",
            "output": "classifier_low_dim/Softmax:0"
        },
        "20180402-114759": {
            "input": "input:0",
            "output": "embeddings:0",
            "phase_train": {
                "phase_train:0": False
            }
        }
    }


    # INITS
    @timer(message="Model load time")
    def __init__(self, filepath=CONFIG_HOME+"/models/ms_celeb_1m.pb", input_name=None, output_name=None, **hyperparams):
        """Initializes FaceNet object

        :param filepath: path to model (default: CONFIG_HOME+"/models/ms_celeb_1m.pb")
        :param input_name: name of input tensor-- only required if using TF-TRT non-default model (default: None)
        :param output_name: name of output tensor-- only required if using TF-TRT non-default model (default: None)
        :param hyperparams: hyperparameters to override FaceNet.HYPERPARAMS

        """

        assert os.path.exists(filepath), "{} not found".format(filepath)

        if ".pb" in filepath:
            self.MODE = "tf-trt"
        elif ".h5" in filepath:
            self.MODE = "keras"
        else:
            raise TypeError("model must be a .h5 or a frozen .pb file")

        if self.MODE == "keras":
            self._keras_init(filepath)
        elif self.MODE == "tf-trt":
            self._trt_init(filepath, input_name, output_name)

        self.__static_db = None  # must be filled in by user
        self.__dynamic_db = {}  # used for real-time database updating (i.e., for visitors)

        self.extra_tensors = []  # extra tensors for feed dict to sess.run

        self.HYPERPARAMS.update(hyperparams)

    def _keras_init(self, filepath):
        """Initializes a Keras model

        :param filepath: path to model (.h5)

        """

        self.facenet = keras.models.load_model(filepath)
        IMG_CONSTANTS["img_size"] = self.facenet.input_shape[1:3]

    def _trt_init(self, filepath, input_name, output_name):
        """Initializes a TF_TRT model

        :param filepath: path to model (.pb)
        :param input_name: name of input tensor
        :param output_name: name of output tensor

        """

        assert tf.test.is_gpu_available(), "TF-TRT mode requires a CUDA-enabled GPU"

        trt_graph = self.get_frozen_graph(filepath)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        tf.import_graph_def(trt_graph, name="")

        self._tensor_init(model_name=filepath, input_name=input_name, output_name=output_name)

        self.facenet = self.sess.graph
        IMG_CONSTANTS["img_size"] = tuple(self.facenet.get_tensor_by_name(self.input_name).get_shape().as_list()[1:3])

    def _tensor_init(self, model_name, input_name, output_name):
        """Initializes tensors (TF-TRT only)

        :param model_name: name of model
        :param input_name: input tensor name
        :param output_name: output tensor name

        """

        self.input_name, self.output_name = None, None

        for model in self.TENSORS:
            if model in model_name:
                self.input_name = self.TENSORS[model]["input"]
                self.output_name = self.TENSORS[model]["output"]

                extra_model_config = self.TENSORS[model]
                extra_model_config.pop("input")
                extra_model_config.pop("output")

                for tensor, value in extra_model_config.items():
                    self.extra_tensors.append({tensor: value})

        if not self.input_name:
            self.input_name = input_name
        elif not self.output_name:
            self.output_name = output_name

        assert self.input_name and self.output_name, "I/O tensors for {} not detected or provided".format(model_name)


    # MUTATORS
    def set_data(self, data):
        """Sets data property

        :param data: new data in form {name: embedding vector, ...}

        """

        assert data is not None, "data must be provided"

        def check_validity(data):
            for key in data.keys():
                assert isinstance(key, str), "data keys must be person names"
                data[key] = np.asarray(data[key])
                is_vector = data[key].ndim <= 2 and (1 in data[key].shape or data[key].ndim == 1)
                assert is_vector, "each data[key] must be a vectorized embedding"
            return data

        self.__static_db = check_validity(data)

        try:
            self._train_knn(knn_types=["static"])
            self.dynamic_knn = None
        except ValueError:
            raise ValueError("Current model incompatible with database")

    def _train_knn(self, knn_types):
        """Trains K-Nearest-Neighbors

        :param knn_types: types of K-NN to train

        """

        def knn_factory(data):
            names, embeddings = zip(*data.items())
            knn = neighbors.KNeighborsClassifier(n_neighbors=len(names) // len(set(names)))
            knn.fit(embeddings, names)
            return knn

        if self.__static_db and "static" in knn_types:
            self.static_knn = knn_factory(self.__static_db)
        if self.__dynamic_db and "dynamic" in knn_types:
            self.dynamic_knn = knn_factory(self.__dynamic_db)


    # RETRIEVERS
    @property
    def data(self):
        """Property for static database

        :returns: self.__static_db

        """

        return self.__static_db

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

    def _make_feed_dict(self, imgs):
        """Makes feed dict for sess.run (TF-TRT only)

        :param imgs: image input
        :returns: feed dict

        """

        feed_dict = {self.input_name: imgs}
        for tensor_dict in self.extra_tensors:
            feed_dict.update(tensor_dict)
        return feed_dict

    def get_embeds(self, data, *args, **kwargs):
        """Gets embedding from various datatypes

        :param data: data dictionary in form {name: embedding vector, ...}
        :param args: data to embed. Can be a key in `data` param, a filepath, or an image array
        :param kwargs: named arguments to self.predict
        :returns: list of embeddings

        """
        embeds = []
        for n in args:
            if isinstance(n, str):
                try:
                    embeds.append(data[n])
                except KeyError:
                    embeds.append(self.predict([n], **kwargs))
            elif not (n.ndim <= 2 and (1 in n.shape or n.ndim == 1)):  # if n is not an embedding vector
                embeds.append(self.predict([n], **kwargs))
            else:
                warnings.warn("{} is not in data or suitable for input into facenet".format(n))

        return embeds if len(embeds) > 1 else embeds[0]

    def predict(self, paths_or_imgs, margin=IMG_CONSTANTS["margin"], faces=None, checkup=False):
        """Low-level predict function (don't use unless developing)

        :param paths_or_imgs: paths or images to predict on
        :param margin: margin for MTCNN face cropping (default: aisecurity.preprocessing.IMG_CONSTANTS["margin"])
        :param faces: pre-detected MTCNN faces (makes `margin` param irrelevant) (default: None)
        :param checkup: whether this is just a call to keep the GPU warm (default: False)
        :returns: L2-normalized embeddings

        """

        l2_normalize = lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), K.epsilon()))

        if self.MODE == "keras":
            predict = lambda imgs: self.facenet.predict(imgs)
        elif self.MODE == "tf-trt":
            output_tensor = self.facenet.get_tensor_by_name(self.output_name)
            predict = lambda imgs: self.sess.run(output_tensor, feed_dict=self._make_feed_dict(imgs))

        cropped_imgs = normalize(crop_faces(paths_or_imgs, margin, faces=faces, checkup=checkup))
        raw_embeddings = predict(cropped_imgs)
        normalized_embeddings = l2_normalize(raw_embeddings)

        return normalized_embeddings


    # FACIAL RECOGNITION HELPER
    @timer(message="Recognition time")
    def _recognize(self, img, metric, db_types=None, **kwargs):
        """Facial recognition under the hood

        :param img: image array
        :param metric: DistMetric object
        :param db_types: database types: "static" and/or "dynamic" (default: None)
        :param kwargs: named arguments to self.get_embeds (will be passed to self.predict)
        :returns: embedding, is recognized (bool), best match from database(s), distance

        """

        assert self.__static_db or self.__dynamic_db, "data must be provided"

        knns, data = [], {}
        if db_types is None or "static" in db_types:
            knns.append(self.static_knn)
            data.update(self.__static_db)
        if db_types is not None and "dynamic" in db_types and self.dynamic_knn and self.__dynamic_db:
            knns.append(self.dynamic_knn)
            data.update(self.__dynamic_db)

        embedding = self.get_embeds(data, img, **kwargs)
        best_matches = []
        for knn in knns:
            pred = knn.predict(embedding)[0]
            best_matches.append((pred, metric(embedding, data[pred])))
        best_match, dist = max(best_matches, key=lambda n: n[1])
        is_recognized = dist <= FaceNet.HYPERPARAMS["alpha"]

        return embedding, is_recognized, best_match, dist

    # FACIAL RECOGNITION
    def recognize(self, img, metric=DistMetric("euclidean"), verbose=True):
        """Facial recognition for a single image

        :param img: image array
        :param metric: DistMetric object (default: DistMetric("euclidean"))
        :param verbose: verbose or not (default: True)
        :returns: is recognized (bool), best match from static database, distance

        """

        _, is_recognized, best_match, dist = self._recognize(img, metric)
        # img can be a path, image, database name key, or embedding

        if verbose:
            if is_recognized:
                print("Your image is a picture of \"{}\": distance of {}".format(best_match, dist))
            else:
                print("Your image is not in the database. The best match is \"{}\" with a distance of ".format(
                    best_match, dist))

        return is_recognized, best_match, dist


    # REAL-TIME FACIAL RECOGNITION HELPER
    async def _real_time_recognize(self, width, height, metric, logging, use_dynamic, use_picam, use_graphics,
                                   use_lcd, use_keypad, framerate, resize, flip):
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
        :returns: number of frames elapsed

        """

        # INITS
        db_types = ["static"]

        if use_dynamic:
            db_types.append("dynamic")
        if logging:
            log.init(flush=True, logging=logging)
            log.server_init()
        if use_lcd:
            lcd.init()
        if use_keypad:
            keypad.init()
        if resize:
            mtcnn_width, mtcnn_height = width * resize, height * resize
        else:
            mtcnn_width, mtcnn_height = width, height

        cap = self.get_video_cap(width, height, picamera=use_picam, framerate=framerate, flip=flip)
        assert cap.isOpened(), "video capture failed to initialize"

        mtcnn = MTCNN(min_face_size=0.5 * (mtcnn_width + mtcnn_height) / 3)
        # face needs to fill at least 1/3 of the frame

        missed_frames = 0
        frames = 0

        last_gpu_checkup = time.time()

        # CAM LOOP
        while True:
            _, frame = cap.read()
            original_frame = frame.copy()

            if resize:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            if use_picam:
                # make sure computation is performed periodically to keep GPU "warm" (i.e., constantly active);
                # otherwise, recognition times can be slow when spaced out by several minutes
                last_gpu_checkup = self.keep_gpu_warm(frame, frames, last_gpu_checkup, use_lcd)

            # using MTCNN to detect faces
            result = mtcnn.detect_faces(frame)

            if result:
                overlay = original_frame.copy()

                person = max(result, key=lambda person: person["confidence"])
                face = person["box"]

                if person["confidence"] < self.HYPERPARAMS["mtcnn_alpha"]:
                    print("Face poorly detected")
                    continue

                # facial recognition
                try:
                    embedding, is_recognized, best_match, dist = self._recognize(frame, metric, db_types, faces=face)
                    print("Distance: {} ({}){}".format(dist, best_match, " !" if not is_recognized else ""))
                except (ValueError, cv2.error) as error:  # error-handling using names is unstable-- change later
                    if "query data dimension" in str(error):
                        raise ValueError("Current model incompatible with database")
                    elif "empty" in str(error):
                        print("Image refresh rate too high")
                    elif "opencv" in str(error):
                        print("Failed to capture frame")
                    else:
                        raise error
                    continue

                # add graphics, lcd, and do logging
                if use_graphics:
                    self.add_graphics(original_frame, overlay, person, width, height, is_recognized, best_match, resize)

                if use_lcd and is_recognized:
                    lcd.PROGRESS_BAR.update(previous_msg="Recognizing...")

                if use_keypad:
                    if is_recognized:
                        run_async_method(keypad.monitor)
                    elif last_best_match != best_match:
                         keypad.CONFIG["continue"] = False
                    # FIXME:
                    #  1. above lines should be changed and use log.current_log instead of making another local var
                    #  2. use of 3 is ambiguous-- add to keypad.CONFIG)
                    #  3. keypad.monitor(0) should be replaced with a reset or flush function if that's what it does

                if logging and frames > 5:  # five frames before logging starts
                    self.log_activity(is_recognized, best_match, use_dynamic, embedding)

                    log.DISTS.append(dist)

            else:
                missed_frames += 1
                if missed_frames > log.THRESHOLDS["missed_frames"]:
                    missed_frames = 0
                    log.flush_current(mode=["known", "unknown"], flush_times=False)
                print("No face detected")

            if use_graphics:
                cv2.imshow("AI Security v0.9a", original_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                # FIXME: doesn't escape when 'q' is pressed-- maybe because of async?
                break

            frames += 1

            await asyncio.sleep(1e-6)

        cap.release()
        cv2.destroyAllWindows()

        return frames


    # REAL-TIME FACIAL RECOGNITION
    def real_time_recognize(self, width=640, height=360, metric=DistMetric("euclidean"), logging="firebase",
                            use_dynamic=False, use_picam=False, use_graphics=True, use_lcd=False, use_keypad=False,
                            framerate=20, resize=None, flip=0):
        """Real-time facial recognition

        :param width: width of frame (only matters if use_graphics is True) (default: 640)
        :param height: height of frame (only matters if use_graphics is True) (default: 360)
        :param metric: DistMetric object or str metric (default: DistMetric("euclidean"), same as "euclidean")
        :param logging: logging type-- None, "firebase", or "mysql" (default: "firebase")
        :param use_dynamic: use dynamic database for visitors or not (default: False)
        :param use_picam: use picamera or not (default: False)
        :param use_graphics: display video feed or not (default: True)
        :param use_lcd: use LCD or not. If LCD isn't connected, will default to LCD simulation and warn (default: False)
        :param use_keypad: use keypad or not. If keypad not connected, will default to False and warn (default: False)
        :param framerate: frame rate, only matters if use_picamera is True (recommended <120) (default: 20)
        :param resize: resize scale (float between 0. and 1.) (default: None)
        :param flip: flip method: +1 = +90º rotation (default: 0)

        """

        assert width > 0 and height > 0, "width and height must be positive integers"
        assert not logging or logging == "mysql" or logging == "firebase", "only mysql and firebase database supported"
        assert 0 < framerate <= 120, "framerate must be between 0 and 120"
        assert resize is None or 0. < resize < 1., "resize must be between 0 and 1"

        if isinstance(metric, str):
            metric = DistMetric(metric)

        run_async_method(
            self._real_time_recognize, width=width, height=height, metric=metric, logging=logging,
            use_dynamic=use_dynamic, use_picam=use_picam, use_graphics=use_graphics, use_lcd=use_lcd,
            use_keypad=use_keypad, framerate=framerate, resize=resize, flip=flip
        )


    # GRAPHICS
    @staticmethod
    def get_video_cap(width, height, picamera, framerate, flip, device=0):
        """Initializes cv2.VideoCapture object

        :param width: width of frame
        :param height: height of frame
        :param picamera: use picamera or not
        :param framerate: framerate, recommended <120
        :param flip: flip method: +1 = +90º rotation (default: 0)
        :param device: VideoCapture will use /dev/video{`device`} (default: 0)
        :returns: cv2.VideoCapture object

        """

        def _gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=640, display_height=360,
                                framerate=20, flip=0):
            return (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12,"
                " framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! video/x-raw, width=(int)%d, height=(int)%d,"
                " format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                % (capture_width, capture_height, framerate, flip, display_width, display_height)
            )

        if picamera:
            return cv2.VideoCapture(
                _gstreamer_pipeline(display_width=width, display_height=height, framerate=framerate, flip=flip),
                cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap

    @staticmethod
    def add_graphics(frame, overlay, person, width, height, is_recognized, best_match, resize):
        """Adds graphics to a frame

        :param frame: frame as array
        :param overlay: overlay as array
        :param person: MTCNN detection dict
        :param width: width of frame
        :param height: height of frame
        :param is_recognized: whether face was recognized or not
        :param best_match: best match from database
        :param resize: resize scale factor, from 0. to 1.

        """

        line_thickness = round(1e-6 * width * height + 1.5)
        radius = round((1e-6 * width * height + 1.5) / 2.)
        font_size = 4.5e-7 * width * height + 0.5
        # works for 6.25e4 pixel video cature to 1e6 pixel video capture

        def get_color(is_recognized, best_match):
            if not is_recognized:
                return 0, 0, 255  # red
            elif "visitor" in best_match:
                return 218, 112, 214  # purple (actually more of an "orchid")
            else:
                return 0, 255, 0  # green

        def add_box_and_label(frame, origin, corner, color, line_thickness, best_match, font_size, thickness):
            cv2.rectangle(frame, origin, corner, color, line_thickness)
            # label box
            cv2.rectangle(frame, (origin[0], corner[1] - 35), corner, color, cv2.FILLED)
            cv2.putText(frame, best_match.replace("_", " ").title(), (origin[0] + 6, corner[1] - 6),
                        cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), thickness)  # white text

        def add_features(overlay, features, radius, color, line_thickness):
            cv2.circle(overlay, (features["left_eye"]), radius, color, line_thickness)
            cv2.circle(overlay, (features["right_eye"]), radius, color, line_thickness)
            cv2.circle(overlay, (features["nose"]), radius, color, line_thickness)
            cv2.circle(overlay, (features["mouth_left"]), radius, color, line_thickness)
            cv2.circle(overlay, (features["mouth_right"]), radius, color, line_thickness)

            cv2.line(overlay, features["left_eye"], features["nose"], color, radius)
            cv2.line(overlay, features["right_eye"], features["nose"], color, radius)
            cv2.line(overlay, features["mouth_left"], features["nose"], color, radius)
            cv2.line(overlay, features["mouth_right"], features["nose"], color, radius)

        features = person["keypoints"]
        x, y, height, width = person["box"]

        if resize:
            scale_factor = 1. / resize

            scale = lambda x: tuple(round(element * scale_factor) for element in x)
            features = {feature: scale(features[feature]) for feature in features}

            scale = lambda *xs: tuple(round(x * scale_factor) for x in xs)
            x, y, height, width = scale(x, y, height, width)

        color = get_color(is_recognized, best_match)

        margin = IMG_CONSTANTS["margin"]
        origin = (x - margin // 2, y - margin // 2)
        corner = (x + height + margin // 2, y + width + margin // 2)

        add_features(overlay, features, radius, color, line_thickness)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        text = best_match if is_recognized else ""
        add_box_and_label(frame, origin, corner, color, line_thickness, text, font_size, thickness=1)


    # DISPLAY
    def show_embeds(self, encrypted=False, single=False):
        """Shows self.data in visual form

        :param encrypted: encrypt data keys (names) before displaying (default: False)
        :param single: show only a single name/embedding pair (default: False)

        """

        assert self.data, "data must be provided to show embeddings"

        def closest_multiples(n):
            if n == 0 or n == 1:
                return n, n
            factors = [((i, int(n / i)), (abs(i - int(n / i)))) for i in range(1, n) if n % i == 0]
            return sorted(factors, key=lambda n: n[1])[0][0]

        if encrypted:
            data = DataEncryption.encrypt_data(self.data, ignore=["embeddings"], decryptable=False)
        else:
            data = self.data

        for person in data:
            embed = np.asarray(data[person])
            embed = embed.reshape(*closest_multiples(embed.shape[0]))

            plt.imshow(embed, cmap="gray")
            try:
                plt.title(person)
            except TypeError:
                warnings.warn("encrypted name cannot be displayed due to presence of non-UTF8-decodable values")
            plt.axis("off")
            plt.show()

            if single and person == list(data.keys())[0]:
                break


    # LOGGING
    def log_activity(self, is_recognized, best_match, use_dynamic, embedding):
        """Logs facial recognition activity

        :param is_recognized: whether face was recognized or not
        :param best_match: best match from database
        :param mode: logging type: "firebase" or "mysql"
        :param use_dynamic: use dynamic database or not
        :param embedding: embedding vector

        """

        cooldown_ok = lambda t: time.time() - t > log.THRESHOLDS["cooldown"]
        mode = lambda d: max(d.keys(), key=lambda key: len(d[key]))

        log.update_current_logs(is_recognized, best_match)

        if log.NUM_RECOGNIZED >= log.THRESHOLDS["num_recognized"] and cooldown_ok(log.LAST_LOGGED):
            if log.get_percent_diff(best_match, log.CURRENT_LOG) <= log.THRESHOLDS["percent_diff"]:
                recognized_person = mode(log.CURRENT_LOG)
                log.log_person(recognized_person, times=log.CURRENT_LOG[recognized_person])

                cprint("Regular activity logged ({})".format(best_match), color="green", attrs=["bold"])

                lcd.add_lcd_display(best_match, log.USE_SERVER)  # will silently fail if lcd not supported

        elif log.NUM_UNKNOWN >= log.THRESHOLDS["num_unknown"] and cooldown_ok(log.UNK_LAST_LOGGED):
            log.log_unknown("<DEPRECATED>")

            cprint("Unknown activity logged", color="red", attrs=["bold"])

            if use_dynamic:
                self.__dynamic_db["visitor_{}".format(len(self.__dynamic_db) + 1)] = embedding.flatten()
                self._train_knn(knn_types=["dynamic"])

                cprint("Visitor activity logged", color="magenta", attrs=["bold"])


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
        if frames == 0 or time.time() - last_gpu_checkup > next_check:
            with HidePrints():
                self._recognize(frame, checkup=True)
            print("Regular computation check")
            last_gpu_checkup = time.time()
            if use_lcd:
                lcd.LCD_DEVICE.clear()
        elif not (time.time() - log.LAST_LOGGED > next_check or time.time() - log.UNK_LAST_LOGGED > next_check):
            last_gpu_checkup = time.time()

        return last_gpu_checkup
