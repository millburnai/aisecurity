import argparse
import platform
import sys
sys.path.insert(1, "../")

if platform.machine() == "arm64":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")

from dataflow.loader import dump_and_embed
from face.detection import FaceDetector
from facenet import FaceNet
from privacy.encryptions import NAMES
from util.distance import DistMetric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="image dir to embed")
    parser.add_argument("--dump_path", help="dump path for embeddings")
    parser.add_argument("--mean", help="use mean or not", action="store_true")
    args = parser.parse_args()

    facenet = FaceNet(data_path=None)
    facenet.dist_metric = DistMetric("cosine", normalize=True)
    facenet.img_norm = "fixed"
    facenet.alpha = 0.3

    detector = FaceDetector("mtcnn", facenet.img_shape)
    dump_and_embed(facenet, args.img_dir, args.dump_path,
                   to_encrypt=NAMES, detector=detector,
                   full_overwrite=True, use_mean=args.mean,
                   verbose=False)
