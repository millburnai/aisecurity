"""

"aisecurity.facenet"

Facial recognition with FaceNet in Keras.

Paper: https://arxiv.org/pdf/1503.03832.pdf

"""

import asyncio
import functools
import json
import os
import time
import warnings

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from keras import backend as K
from mtcnn.mtcnn import MTCNN
from skimage.transform import resize
from sklearn import neighbors
from termcolor import cprint

from aisecurity.extras.paths import CONFIG_HOME
from aisecurity.logs import log
from aisecurity.security.encryptions import DataEncryption


# DECORATORS
def timer(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(time.time() - start, 3)))
            return result

        return _func

    return _timer


# FACENET
class FaceNet(object):

    # HYPERPARAMETERS
    HYPERPARAMS = {
        "alpha": 1.0,
        "margin": 10,
        "clear": 0.5,
        "update_alpha": 5
    }

    # INITS
    @timer(message="Model load time")
    def __init__(self, filepath=CONFIG_HOME + "/models/ms_celeb_1m.h5"):
        assert os.path.exists(filepath), "{} not found".format(filepath)
        self.facenet = keras.models.load_model(filepath)

        self.__static_data = None  # must be filled in by user
        self.__dynamic_data = None  # used for real-time database updating (i.e., for visitors)

    # MUTATORS
    def set_data(self, data):
        assert data is not None, "data must be provided"

        def check_validity(data):
            for key in data.keys():
                assert isinstance(key, str), "data keys must be person names"
                data[key] = np.asarray(data[key])
                is_vector = data[key].ndim <= 2 and (1 in data[key].shape or data[key].ndim == 1)
                assert is_vector, "each data[key] must be a vectorized embedding"
            return data

        self.__static_data = check_validity(data)

        self._train_knn(knn_types=["static", "dynamic"])

    def _train_knn(self, knn_types):
        def knn_factory(data):
            knn_label_dict, embeddings = [], []
            for person in data.keys():
                knn_label_dict.append(person)
                embeddings.append(data[person])
            knn = neighbors.KNeighborsClassifier(n_neighbors=len(knn_label_dict) // len(set(knn_label_dict)))
            knn.fit(embeddings, knn_label_dict)
            return knn

        if "static" in knn_types and self.__static_data:
            self.static_knn = knn_factory(self.__static_data)
        if "dynamic" in knn_types and self.__dynamic_data:
            self.dynamic_knn = knn_factory(self.__dynamic_data)


    # RETRIEVERS
    @property
    def data(self):
        return self.__static_data

    def get_embeds(self, data, *args, **kwargs):
        embeds = []
        for n in args:
            if isinstance(n, str):
                try:
                    n = data[n]
                except KeyError:
                    n = self.predict([n], margin=self.HYPERPARAMS["margin"], **kwargs)
            elif not (n.ndim <= 2 and (1 in n.shape or n.ndim == 1)):  # n must be a vector
                n = self.predict([n], margin=self.HYPERPARAMS["margin"], **kwargs)
            embeds.append(n)
        return tuple(embeds) if len(embeds) > 1 else embeds[0]

    def predict(self, paths_or_imgs, *args, **kwargs):
        return Preprocessing.embed(self.facenet, paths_or_imgs, *args, **kwargs)

    # FACIAL RECOGNITION HELPER
    @timer(message="Recognition time")
    def _recognize(self, img, faces=None, data_types=None):
        assert self.__static_data or self.__dynamic_data, "data must be provided"

        knns, data = [], {}
        if data_types is None or "static" in data_types:
            knns.append(self.static_knn)
            data.update(self.__static_data)
        if "dynamic" in data_types:
            knns.append(self.dynamic_knn)
            data.update(self.__dynamic_data)

        embedding = self.get_embeds(data, img, faces=faces)
        best_matches = []
        for knn in reversed(knns):
            pred = knn.predict(embedding)[0]
            best_matches.append((pred, np.linalg.norm(embedding - data[pred])))

        best_match, l2_dist = sorted(best_matches, key=lambda n: n[1])

        return embedding, l2_dist <= FaceNet.HYPERPARAMS["alpha"], best_match, l2_dist

    # FACIAL RECOGNITION
    def recognize(self, img, verbose=True):
        # img can be a path, image, database name, or embedding
        _, is_recognized, best_match, l2_dist = self._recognize(img)

        if verbose:
            if is_recognized:
                print("Your image is a picture of \"{}\": L2 distance of {}".format(best_match, l2_dist))
            else:
                print("Your image is not in the database. The best match is \"{}\" with an L2 distance of ".format(
                    best_match, l2_dist))

        return is_recognized, best_match, l2_dist

    # REAL-TIME FACIAL RECOGNITION HELPER
    async def _real_time_recognize(self, width, height, use_log, adaptive_alpha, use_dynamic):
        if use_log:
            log.init(flush=True)
        if use_dynamic:
            data_types = ["static", "dynamic"]
        else:
            data_types = ["static"]

        detector = MTCNN()
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        missed_frames = 0
        l2_dists = []

        while True:
            _, frame = cap.read()
            result = detector.detect_faces(frame)

            if result:
                overlay = frame.copy()

                for person in result:
                    # using MTCNN to detect faces
                    face = person["box"]

                    # facial recognition
                    try:
                        embedding, is_recognized, best_match, l2_dist = self._recognize(frame, face, data_types)
                        print("L2 distance: {} ({})".format(l2_dist, best_match))
                    except ValueError:
                        print("Image refresh rate too high")
                        continue

                    if use_dynamic:
                        if not is_recognized and person["confidence"] >= self.HYPERPARAMS["mtcnn_alpha"]:
                            self.update_dynamic(embedding)

                    self.add_graphics(frame, overlay, person, width, height, is_recognized, best_match)

                    # log activity
                    if use_log:
                        self.log_activity(is_recognized, best_match, frame, log_unknown=True)

                    # adaptive recognition threshold
                    if is_recognized and adaptive_alpha:
                        l2_dists.append(l2_dist)
                        l2_dists = self.update_alpha(l2_dists)

            else:
                missed_frames += 1
                if missed_frames > log.THRESHOLDS["missed_frames"]:
                    missed_frames = 0
                    log.flush_current()
                print("No face detected")

            cv2.imshow("CSII AI facial recognition v0.1", frame)

            await asyncio.sleep(K.epsilon())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # REAL-TIME FACIAL RECOGNITION
    def real_time_recognize(self, width=500, height=250, use_log=True):

        async def async_helper(recognize_func, *args, **kwargs):
            await recognize_func(*args, **kwargs)

        loop = asyncio.new_event_loop()
        task = loop.create_task(async_helper(self._real_time_recognize, width, height, use_log, adaptive_alpha=True,
                                             use_dynamic=False))
        loop.run_until_complete(task)

    # GRAPHICS
    def add_graphics(self, frame, overlay, person, width, height, is_recognized, best_match):
        line_thickness = round(1e-6 * width * height + 1.5)
        radius = round((1e-6 * width * height + 1.5) / 2.)
        font_size = 4.5e-7 * width * height + 0.5
        # works for 6.25e4 pixel video cature to 1e6 pixel video capture

        def get_color(is_recognized, best_match):
            if not is_recognized:
                return 0, 0, 255  # red
            elif "visitor" in best_match:
                return 128, 0, 128  # purple
            else:
                return 0, 255, 0  # green

        def add_box_and_label(frame, corner, box, color, line_thickness, best_match, font_size, thickness):
            cv2.rectangle(frame, corner, box, color, line_thickness)
            cv2.putText(frame, best_match, corner, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)

        def add_key_points(overlay, key_points, radius, color, line_thickness):
            cv2.circle(overlay, (key_points["left_eye"]), radius, color, line_thickness)
            cv2.circle(overlay, (key_points["right_eye"]), radius, color, line_thickness)
            cv2.circle(overlay, (key_points["nose"]), radius, color, line_thickness)
            cv2.circle(overlay, (key_points["mouth_left"]), radius, color, line_thickness)
            cv2.circle(overlay, (key_points["mouth_right"]), radius, color, line_thickness)

            cv2.line(overlay, key_points["left_eye"], key_points["nose"], color, radius)
            cv2.line(overlay, key_points["right_eye"], key_points["nose"], color, radius)
            cv2.line(overlay, key_points["mouth_left"], key_points["nose"], color, radius)
            cv2.line(overlay, key_points["mouth_right"], key_points["nose"], color, radius)

        key_points = person["keypoints"]
        x, y, height, width = person["box"]

        color = get_color(is_recognized, best_match)

        margin = self.HYPERPARAMS["margin"]
        corner = (x - margin // 2, y - margin // 2)
        box = (x + height + margin // 2, y + width + margin // 2)

        add_key_points(overlay, key_points, radius, color, line_thickness)
        clear = self.HYPERPARAMS["clear"]
        cv2.addWeighted(overlay, 1.0 - clear, frame, clear, 0, frame)

        text = best_match if is_recognized else ""
        add_box_and_label(frame, corner, box, color, line_thickness, text, font_size, thickness=1)

    # DISPLAY
    def show_embeds(self, encrypted=False, single=False):

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
                warnings.warn("encrypted data cannot be displayed due to presence of non-UTF8-decodable values")
            plt.axis("off")
            plt.show()

            if single and person == list(data.keys())[0]:
                break

    # LOGGING
    @staticmethod
    def log_activity(is_recognized, best_match, frame, log_unknown=True):
        cooldown_ok = lambda t: time.time() - t > log.THRESHOLDS["cooldown"]

        def get_mode(d):
            max_key = list(d.keys())[0]
            for key in d:
                if len(d[key]) > len(d[max_key]):
                    max_key = key
            return max_key

        log.update_current_logs(is_recognized, best_match)

        if log.num_recognized >= log.THRESHOLDS["num_recognized"] and cooldown_ok(log.last_logged):
            if log.get_percent_diff(best_match) <= log.THRESHOLDS["percent_diff"]:
                recognized_person = get_mode(log.current_log)
                log.log_person(recognized_person, times=log.current_log[recognized_person])
                cprint("Regular activity logged", color="green", attrs=["bold"])

        if log_unknown and log.num_unknown >= log.THRESHOLDS["num_unknown"] and cooldown_ok(log.unk_last_logged):
            path = CONFIG_HOME + "/database/unknown/{}.jpg".format(len(os.listdir(CONFIG_HOME + "/database/unknown")))
            log.log_unknown(path)
            # recording unknown images is deprecated and will be removed/changed later
            cv2.imwrite(path, frame)
            cprint("Unknown activity logged", color="red", attrs=["bold"])

    def update_dynamic(self, embedding):
        raise NotImplementedError()

    # ADAPTIVE ALPHA
    def update_alpha(self, l2_dists):
        if len(l2_dists) % round(self.HYPERPARAMS["update_alpha"]) == 0:
            updated = 0.9 * self.HYPERPARAMS["alpha"] + 0.1 * (sum(l2_dists) / len(l2_dists) + 0.3)
            # alpha is a weighted average of the previous alpha and the new alpha
            self.HYPERPARAMS["update_alpha"] *= 1 + (updated / self.HYPERPARAMS["alpha"])
            # update alpha changes proportionally to the magnitude of the update
            self.HYPERPARAMS["alpha"] = updated
            l2_dists = []
        return l2_dists


# IMAGE PREPROCESSING
class Preprocessing(object):

    # HYPERPARAMETERS
    IMG_SIZE = 160

    @staticmethod
    def whiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError("x must have either 3 or 4 dimensions")

        std_adj = np.maximum(np.std(x, axis=axis, keepdims=True), 1.0 / np.sqrt(size))
        whitened = (x - np.mean(x, axis=axis, keepdims=True)) / std_adj
        return whitened

    @staticmethod
    def align_imgs(paths_or_imgs, margin, faces=None):
        if not faces:
            detector = MTCNN()

        def align_img(path_or_img, faces=None):
            try:
                img = imread(path_or_img)
            except OSError:  # if img is embedding
                img = path_or_img

            if not faces:
                found = detector.detect_faces(img)
                assert len(found) != 0, "face was not found in {}".format(path_or_img)
                faces = found[0]["box"]

            x, y, width, height = faces
            cropped = img[y - margin // 2:y + height + margin // 2, x - margin // 2:x + width + margin // 2, :]
            resized = resize(cropped, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE))
            return resized

        return np.array([align_img(path_or_img, faces=faces) for path_or_img in paths_or_imgs])

    @staticmethod
    def embed(facenet, paths_or_imgs, margin=15, batch_size=1, faces=None):
        aligned_imgs = Preprocessing.whiten(Preprocessing.align_imgs(paths_or_imgs, margin, faces=faces))
        raw_embeddings = facenet.predict(aligned_imgs, batch_size=batch_size)
        l2_normalize = lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), K.epsilon()))
        normalized_embeddings = l2_normalize(raw_embeddings)
        return normalized_embeddings

    @staticmethod
    @timer(message="Data preprocessing time")
    def load(facenet, img_dir, people=None):
        if people is None:
            people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
        data = {person: Preprocessing.embed(facenet, img_dir + person) for person in people}
        return data

    @staticmethod
    @timer(message="Data dumping time")
    def dump_embeds(facenet, img_dir, dump_path, retrieve_path=None, full_overwrite=False, ignore_encrypt=None):

        if ignore_encrypt == "all":
            ignore_encrypt = ["names", "embeddings"]
        elif ignore_encrypt is not None:
            ignore_encrypt = [ignore_encrypt]

        if not full_overwrite:
            people = [f for f in os.listdir(img_dir) if not f.endswith(".DS_Store") and not f.endswith(".json")]
            old_embeds = Preprocessing.retrieve_embeds(retrieve_path if retrieve_path is not None else dump_path)

            new_people = [person for person in people if person not in old_embeds.keys()]
            new_embeds = Preprocessing.load(facenet.facenet, img_dir, people=new_people)

            embeds_dict = {**old_embeds, **new_embeds}  # combining dicts and overwriting any duplicates with new_embeds
        else:
            embeds_dict = Preprocessing.load(facenet.facenet, img_dir)

        encrypted_data = DataEncryption.encrypt_data(embeds_dict, ignore=ignore_encrypt)

        with open(dump_path, "w+") as json_file:
            json.dump(encrypted_data, json_file, indent=4, ensure_ascii=False)

    @staticmethod
    @timer(message="Data retrieval time")
    def retrieve_embeds(path):
        with open(path, "r") as json_file:
            data = json.load(json_file)

        try:  # default case: names are encoded, embeddings aren't
            return DataEncryption.decrypt_data(data, ignore=["embeddings"])
        except UnicodeDecodeError:  # if names cannot be decoded
            try:
                return DataEncryption.decrypt_data(data, ignore=["names"])
            except TypeError:  # if embeddings cannot be decoded
                return data
