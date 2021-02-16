import argparse
import sys
sys.path.insert(1, "../")

from dataflow.loader import dump_and_embed
from face.detection import FaceDetector
from facenet import FaceNet
from utils.distance import DistMetric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", "image dir to embed")
    parser.add_argument("--dump_path", "dump path for embeddings")
    args = parser.parse_args()

    facenet = FaceNet(data_path=None, allow_gpu_growth=True)
    facenet.dist_metric = DistMetric("cosine", normalize=True)
    facenet.img_norm = "fixed"

    detector = FaceDetector("trt-mtcnn")
    dump_and_embed(facenet, args.img_dir, args.dump_path,
                   encrypt="names", detector=detector)
