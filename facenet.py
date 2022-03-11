"""Facial recognition with FaceNet in Keras, TensorFlow, or TensorRT.
"""

from copy import copy
import json
import os
from timeit import default_timer as timer
import threading

import cv2
import numpy as np
from sklearn import neighbors, svm
from termcolor import colored

try:
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda  # noqa
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] '{e}'. Ignore if GPU is not set up")

try:
    import tensorrt as trt  # noqa
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] '{e}'. Ignore if GPU is not set up")

from util.detection import FaceDetector, is_looking
from util.distance import DistMetric
from util.loader import (
    print_time,
    screen_data,
    strip_id,
    retrieve_embeds,
    get_frozen_graph,
)
from util.common import DB_LOB, DEFAULT_MODEL, EMBED_KEY_PATH, NAME_KEY_PATH
from util.pbar import ProgressBar
from util.visuals import Camera, GraphicsRenderer
from util.log import Logger


class FaceNet:
    """Class implementation of FaceNet"""

    @print_time("model load time")
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        data_path=DB_LOB,
        input_name="input",
        output_name="embeddings",
        input_shape=(160, 160),
        classifier="svm",
        gpu_alloc=False,
    ):
        """Initializes FaceNet object
        :param model_path: path to model (default: utils.paths.DEFAULT_MODEL)
        :param data_path: path to data (default: utils.paths.DB_LOB)
        :param input_name: input - TF mode only (default: "input:0")
        :param output_name: output - TF mode only (default: "embeddings:0")
        :param input_shape: input shape in HW (default: (160, 160))
        :param classifier: classifier type (default: 'svm')
        :param gpu_alloc: allow GPU growth (default: False)
        """

        assert os.path.exists(model_path), f"{model_path} not found"
        assert not data_path or os.path.exists(data_path), f"{data_path} not found"

        if gpu_alloc:
            import tensorflow as tf  # noqa

            try:
                gpus = tf.config.experimental.list_physical_devices("GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as err:
                print(err)

        if ".h5" in model_path:
            self._keras_init(model_path)
        elif ".tflite" in model_path:
            self._tflite_init(model_path)
        elif ".pb" in model_path:
            self._tf_init(
                model_path, input_name + ":0", output_name + ":0", input_shape
            )
        elif ".engine" in model_path:
            self._trt_init(model_path, input_shape)
        else:
            raise TypeError("model must be an .h5, .pb, or .engine file")
        print(f"[DEBUG] inference backend is {self.mode}")

        self._db = {}
        self._db_threshold = {}
        self.classifier = None
        self.classifier_type = classifier

        if data_path:
            self.set_data(
                *retrieve_embeds(
                    data_path, name_keys=NAME_KEY_PATH, embedding_keys=EMBED_KEY_PATH
                )
            )
        else:
            print("[DEBUG] data not set. Set it manually with set_data")

    @property
    def data(self):
        """Property for static database of embeddings
        :returns: self._db
        """

        return self._db

    @property
    def metadata(self):
        return {
            "metric": self.dist_metric.metric,
            "normalize": self.dist_metric.normalize,
            "alpha": self.alpha,
            "img_norm": self.img_norm,
        }

    @property
    def data_threshold(self):
        """Property for static database of thresholds
        :returns: self._db_threshold
        """
        return self._db_threshold

    def _keras_init(self, filepath):
        """Initializes a Keras model
        :param filepath: path to model (.h5)
        """

        import tensorflow.compat.v1 as tf  # noqa

        self.mode = "keras"
        self.facenet = tf.keras.models.load_model(filepath)
        self.img_shape = self.facenet.input_shape[1:3]

    def _tflite_init(self, filepath):
        """Initializes a tflite model interpreter
        :param filepath: path to model (.tflite)
        """

        import tensorflow.compat.v1 as tf  # noqa

        self.mode = "tflite"
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

        self.mode = "tf"

        self.input_name = input_name
        self.output_name = output_name
        self.img_shape = input_shape

        graph_def = get_frozen_graph(filepath)
        self.sess = tf.keras.backend.get_session()

        tf.import_graph_def(graph_def, name="")
        self.facenet = self.sess.graph

    def _trt_init(self, filepath, input_shape):
        """TensorRT initialization
        :param filepath: path to serialized engine
        :param input_shape: input shape
        """

        self.mode = "trt"
        try:
            self.dev_ctx = cuda.Device(0).make_context()
            self.stream = cuda.Stream()
            trt_logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(trt_logger)

            with open(filepath, "rb") as model:
                self.facenet = runtime.deserialize_cuda_engine(model.read())

            self.context = self.facenet.create_execution_context()

            self.h_input = cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(0)), dtype=np.float32
            )
            self.h_output = cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(1)), dtype=np.float32
            )

            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            print("here")

        except NameError:
            raise ValueError("trt mode requested but not available")

        self.img_shape = input_shape

    def add_entry(self, person, embeddings, train_classifier=True):
        """Adds entry (person, embeddings) to database
        :param person: new entry
        :param embeddings: new entry's list of embeddings
        :param train_classifier: train classifier (default: True)
        """
        screen_data(person, embeddings)

        embeds = np.array(embeddings).reshape(len(embeddings), -1)
        self._db[person] = embeds
        self._db_threshold[person] = 0

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

        self.dist_metric = DistMetric(
            self.data_cfg["metric"],
            self.data_cfg["normalize"],
            self.data_cfg.get("mean"),
        )
        self.alpha = self.data_cfg["alpha"]
        self.img_norm = self.data_cfg["img_norm"]

        if data:
            for person, embed in data.items():
                self.add_entry(person, embed, train_classifier=False)
            self.apply_thresholds()
            self._train_classifier()

    def _train_classifier(self):
        """Trains person classifier"""
        try:
            if self.classifier_type == "svm":
                self.classifier = svm.SVC(kernel="linear")
            elif self.classifier_type == "knn":
                self.classifier = neighbors.KNeighborsClassifier()

            embeds = np.squeeze(list(self.data.values()), axis=1)
            self.classifier.fit(embeds, self._stripped_names)

        except (AttributeError, ValueError):
            raise ValueError("Current model incompatible with database")

    def normalize(self, imgs):
        if self.img_norm == "per_image":
            # linearly scales x to have mean of 0, variance of 1
            std_adj = np.std(imgs, axis=(1, 2, 3), keepdims=True)
            std_adj = np.maximum(std_adj, 1.0 / np.sqrt(imgs.size / len(imgs)))
            mean = np.mean(imgs, axis=(1, 2, 3), keepdims=True)
            return (imgs - mean) / std_adj
        elif self.img_norm == "fixed":
            # scales x to [-1, 1]
            return (imgs - 127.5) / 128.0
        else:
            return imgs

    def embed(self, imgs):
        """Embeds cropped face
        :param imgs: list of cropped faces with shape (b, h, w, 3)
        :returns: embedding as array with shape (1, -1)
        """

        if self.mode == "keras":
            embeds = self.facenet.predict(imgs, batch_size=len(imgs))
        elif self.mode == "tf":
            out = self.facenet.get_tensor_by_name(self.output_name)
            embeds = self.sess.run(out, feed_dict={self.input_name: imgs})
        elif self.mode == "tflite":
            imgs = imgs.astype(np.float32)
            self.facenet.set_tensor(self.input_details[0]["index"], imgs)
            self.facenet.invoke()
            embeds = self.facenet.get_tensor(self.output_details[0]["index"])
        else:
            if len(imgs) != 1:
                raise NotImplementedError("trt batch not yet supported")
            threading.Thread.__init__(self)
            self.dev_ctx.push()

            np.copyto(self.h_input, imgs.astype(np.float32).ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async(
                batch_size=1,
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle,
            )
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            self.dev_ctx.pop()

            embeds = np.copy(self.h_output)

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

        cropped_faces, face_coords = detector.crop_face(img, margin, flip, verbose)
        if cropped_faces is None:
            if verbose:
                print("No face detected")
            return None, None

        start = timer()
        

        normalized = self.normalize(np.array(cropped_faces))
        embeds = self.embed(normalized)
        embeds = self.dist_metric.apply_norms(embeds, batch=True)

        if verbose:
            elapsed = round(1000.0 * (timer() - start), 2)
            time = colored(f"{elapsed} ms", attrs=["bold"])
            vecs = f"{len(embeds)} vector{'s' if len(embeds) > 1 else ''}"
            print(f"Embedding time ({vecs}): {time}")

        s = self.find_similar_embedding(embeds)
        print(s, list(self.data.keys())[s])

        return embeds, face_coords

    def recognize(self, img, *args, verbose=True, **kwargs):
        """Facial recognition
        :param img: image array in BGR mode
        :param args: will be passed to self.predict
        :param verbose: verbose or not (default: True)
        :param kwargs: will be passed to self.predict
        :returns: face, is recognized, best match, time elapsed
        """
        start = timer()

        is_recognized = None
        best_match = None
        face = None

        try:
            embeds, face = self.predict(img, *args, **kwargs, verbose=verbose)
            if embeds is not None:
                best_match = self.classifier.predict(embeds)[0]

                nearest = self._stripped_db[best_match]

                dists = self.dist_metric.distance(embeds, nearest, True)
                dist = np.average(dists)
                is_recognized = dist <= self.alpha

                if verbose and dist:
                    info = colored(
                        f"{round(dist, 4)} ({best_match})",
                        color="green" if is_recognized else "red",
                    )
                    print(f"{self.dist_metric}: {info}")

        except (ValueError, cv2.error) as error:
            incompatible = "query data dimension"
            if isinstance(error, ValueError) and incompatible in str(error):
                raise ValueError("Current model incompatible with database")
            elif isinstance(error, cv2.error) and "resize" in str(error):
                print("Frame capture failed")
            else:
                raise error

        elapsed = round(1000.0 * (timer() - start), 4)
        return face, is_recognized, best_match, elapsed

    def real_time_recognize(
        self,
        width=640,
        height=360,
        resize=1.0,
        detector="mtcnn",
        flip=False,
        graphics=True,
        socket=None,
        mtcnn_stride=1,
    ):
        """Real-time facial recognition
        :param width: width of frame (default: 640)
        :param height: height of frame (default: 360)
        :param resize: resize scale (default: 1. = no resize)
        :param detector: face detector type (default: "mtcnn")
        :param flip: whether to flip horizontally or not (default: False)
        :param graphics: whether or not to use graphics (default: True)
        :param socket: socket (dev) (default: None)
        :param mtcnn_stride: stride frame stride (default: 1)
        """

        assert self._db, "data must be provided"
        assert 0.0 <= resize <= 1.0, "resize must be in [0., 1.]"

        graphics_controller = GraphicsRenderer(width, height, resize)
        logger = Logger(frame_limit=10, frame_threshold=5)
        pbar = ProgressBar(logger, ws=socket)
        cap = Camera()
        detector = FaceDetector(
            detector, self.img_shape, min_face_size=240, stride=mtcnn_stride
        )

        while True:
            _, frame = cap.read()
            cframe = frame.copy()

            # resize frame
            if resize != 1:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

            # facial detection and recognition
            info = self.recognize(frame, detector, flip=flip)
            face, is_recognized, best_match, elapsed = info

            # logging and socket
            if is_recognized and is_looking(face):
                log_result = logger.log(best_match)
                pbar.update(end=log_result is not None)
                if log_result and socket:
                    socket.send(json.dumps({"best_match": best_match}))

            # graphics
            if graphics:
                graphics_controller.add_graphics(cframe, *info)
                cv2.imshow("AI Security v2021.0.1", cframe)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def compute_similarity(self, embedding1, embedding2) -> float:
        """
        https://arxiv.org/pdf/1810.11160.pdf

        In Figure 1, the letters (A, B, C) are representing
        different people and person C is being inputted into
        the facial recognition.

        In Figure 2, the person is being added to the database
        with their name and embedding registered. All embeddings
        are extracted from other people and are being compared
        using the dot product with the inputted person's embedding
        and the other people's embedding inputted into the database.
        The dot product results are being compared.

        t = ITERATOR & total amount of embeddings
        In Figure 2, F4 is the max Ft can go.

        FT = INPUT PERSON's embedding
        Fv = v person's embedding
        S(T, v) = dot(FT, Fv)

        F(fancy S korean character) embedding of the facial image without identity
        Fu is themost similar embedding

        Questions:
        - If the database is empty, how do you get the thresholds?
        - If you found a new similarity, do you keep or replace the
        old embedding and threshold?
        
        1 ... n = row
        1, ..., n = column
        """
        return np.dot(
            np.squeeze(np.asarray(embedding1)), np.squeeze(np.asarray(embedding2))
        )

    def find_threshold(self, person) -> float:
        """This might be correct, not 100% sure.
        """
        embedding = self.data[person]
        compares = []
        people = copy(self.data)
        del people[person]
        people_thresholds = people.values()
        for x in people_thresholds:
            s = self.compute_similarity(embedding, x)
            compares.append(s)
        
        return np.max(np.std(compares))

    def apply_thresholds(self) -> None:
        people = list(self.data.keys())
        i = 0
        for person in people:
            thresholds = [0]
            if i != 0:
                thresholds = [0, self.find_threshold(people[i-1])]
            
            for j in range(len(people)):
                person_name1 = people[j].split("-")[0]
                person_name2 = person.split("-")[0]
                if person_name1 != person_name2:
                    thresholds.append(self.compute_similarity(self.data[people[j]], self.data[person]))

            self._db_threshold[person] = np.max(thresholds)
            i += 1

    def find_similar_embedding(self, embedding) -> int:
        """Returns index of similar embedding for self.data
        """
        compares = []
        for x in self.data.values():
            s = self.compute_similarity(embedding, x)
            compares.append(s)
        return np.argmax(compares)

    def is_intruder(self, embedding) -> bool:
        simliar_embedding = self.find_similar_embedding(embedding)
        
        