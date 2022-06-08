"""MTCNN face detection.
"""

from functools import partial
import sys
from timeit import default_timer as timer
from typing import Tuple, List

import cv2
from termcolor import colored

sys.path.insert(1, "../")
from util.common import CONFIG_HOME


def is_looking(face):
    pts = face["keypoints"]
    eye_diff = abs(pts["right_eye"][0] - pts["left_eye"][0])
    x, y, w, h = face["box"]

    ratio = eye_diff / w
    print(f"ratio: {ratio}")
    return ratio > 0.4


def get_mtcnn(
    mtcnn_path,
    min_size: float = 40.0,
    factor: float = 0.709,
    thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
):
    import tensorflow.compat.v1 as tf  # noqa

    def mtcnn(img, mtcnn_path, min_size, factor, thresholds):
        with tf.gfile.GFile(mtcnn_path, "rb") as f:
            graph_def = tf.GraphDef.FromString(f.read())

        prob, landmarks, box = tf.import_graph_def(
            graph_def,
            input_map={
                "input:0": img,
                "min_size:0": min_size,
                "thresholds:0": thresholds,
                "factor:0": factor,
            },
            return_elements=["prob:0", "landmarks:0", "box:0"],
        )

        return box, prob, landmarks

    return partial(
        mtcnn,
        mtcnn_path=mtcnn_path,
        min_size=min_size,
        factor=factor,
        thresholds=thresholds,
    )


class FaceDetector:
    def __init__(
        self,
        mode,
        img_shape: Tuple[int, int] = (160, 160),
        alpha: float = 0.8,
        stride: int = 1,
        min_face_size: int = 40,
    ) -> None:
        assert mode in ("mtcnn", "trt-mtcnn"), f"{mode} not supported"

        self.mode = mode
        self.alpha = alpha
        self.img_shape = tuple(img_shape)
        self.stride = stride
        self.min_face_size = min_face_size
        self.frame_ct = 0
        self._cached_result = None

        if "trt-mtcnn" in mode:
            sys.path.insert(1, "../util/trt_mtcnn_plugin")
            from util.trt_mtcnn_plugin.trt_mtcnn import TrtMTCNNWrapper  # noqa

            engine_paths = [
                f"../util/trt_mtcnn_plugin/mtcnn/det{i+1}.engine" for i in range(3)
            ]
            self.trt_mtcnn = TrtMTCNNWrapper(*engine_paths)

        if "mtcnn" in mode.replace("trt-mtcnn", ""):
            import tensorflow.compat.v1 as tf  # noqa

            assert tf.executing_eagerly(), (
                "[internal] launch failed, tf not eager."
                "Check that tensorflow>=2.3 and that eager exec enabled"
            )

            mpath = CONFIG_HOME + "/models/mtcnn.pb"
            self.mtcnn = tf.wrap_function(
                get_mtcnn(mpath, min_size=float(self.min_face_size)),
                [tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32)],
            )

    def detect_faces(self, img) -> List[dict]:
        self.frame_ct += 1
        result = []

        skip_frame = self.stride != 1 and self.frame_ct % self.stride != 0
        if skip_frame and self._cached_result:
            result = self._cached_result

        elif "trt-mtcnn" in self.mode:
            result = self.trt_mtcnn.detect_faces(img)

        elif "mtcnn" in self.mode.replace("trt-mtcnn", ""):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bboxes, scores, landmarks = map(lambda x: x.numpy(), self.mtcnn(img))

            for face, score, pts in zip(bboxes, scores, landmarks):
                x, y = int(face[1]), int(face[0])
                width, height = int(face[3] - x), int(face[2] - y)

                pts = list(map(int, pts))
                result.append(
                    {
                        "box": (x, y, width, height),
                        "confidence": score,
                        "keypoints": {
                            "left_eye": (pts[5], pts[0]),
                            "right_eye": (pts[6], pts[1]),
                            "nose": (pts[7], pts[2]),
                            "mouth_left": (pts[8], pts[3]),
                            "mouth_right": (pts[9], pts[4]),
                        },
                    }
                )

        self._cached_result = result
        return result

    def crop_face(self, img_bgr, margin, flip: bool = False, verbose: bool = True):
        start = timer()
        resized_faces, face = None, None

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.detect_faces(img)

        if len(result) != 0:
#            face = max(result, key=lambda person: person["confidence"])
            face = max(result, key=lambda person: person["box"][-1] * person["box"][-2])

            if face["confidence"] >= self.alpha:
                x, y, width, height = face["box"]
                img = img[
                    y - margin // 2 : y + height + margin // 2,
                    x - margin // 2 : x + width + margin // 2,
                    :,
                ]
                resized_faces = [cv2.resize(img, self.img_shape)]

                if flip:
                    flipped = cv2.flip(resized_faces[0], 1)
                    resized_faces.append(flipped)

                if verbose:
                    elapsed = round(1000.0 * (timer() - start), 2)
                    time = colored(f"{elapsed} ms", attrs=["bold"])
                    print(f"Detection time ({self.mode}): " + time)

            elif verbose:
                confidence = round(face["confidence"] * 100, 2)
                print(f"{confidence}% detect confidence (too low)")

        elif verbose:
            print("No face detected")

        return resized_faces, face
