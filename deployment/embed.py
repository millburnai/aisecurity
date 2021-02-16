import argparse
import sys
sys.path.insert(1, "../")

from dataflow.loader import dump_and_embed
from face.detection import FaceDetector
from facenet import FaceNet
from privacy.encryptions import NAMES
from utils.distance import DistMetric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="image dir to embed")
    parser.add_argument("--dump_path", help="dump path for embeddings")
    args = parser.parse_args()

    facenet = FaceNet(data_path=None, allow_gpu_growth=True)
    facenet.dist_metric = DistMetric("cosine", normalize=True)
    facenet.img_norm = "fixed"
    facenet.alpha = 0.75

    detector = FaceDetector("trt-mtcnn")
    dump_and_embed(facenet, args.img_dir, args.dump_path,
                   to_encrypt=NAMES, detector=detector, full_overwrite=True)
