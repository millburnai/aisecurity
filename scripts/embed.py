import argparse
import sys

sys.path.insert(1, "../")

from facenet import FaceNet
from util.loader import dump_and_embed
from util.detection import FaceDetector
from util.encryptions import NAMES, ALL
from util.distance import DistMetric
from util.common import ON_CUDA, NAME_KEY_PATH, EMBED_KEY_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", help="image dir to embed")
    parser.add_argument("dump_path", help="dump path for embeddings")
    parser.add_argument("--mean", help="use mean or not", action="store_true")
    parser.add_argument("--name_keys", help="name keys path",
                        default=NAME_KEY_PATH)
    parser.add_argument("--embed_keys", help="embed keys path",
                        default=EMBED_KEY_PATH)
    args = parser.parse_args()

    facenet = FaceNet(data_path=None)
    facenet.dist_metric = DistMetric("cosine", normalize=True)
    facenet.img_norm = "fixed"
    facenet.alpha = 0.33

    detector = FaceDetector("trt-mtcnn" if ON_CUDA else "mtcnn", facenet.img_shape)
    no_faces = dump_and_embed(
        facenet,
        args.img_dir,
        args.dump_path,
        to_encrypt=ALL,
        full_overwrite=True,
        use_mean=args.mean,
        load_kwargs=dict(
            detector=detector,
            verbose=False
        ),
        encrypt_kwargs=dict(
            name_keys=args.name_keys,
            embedding_keys=args.embed_keys
        )
    )
    print(f"[DEBUG] faces not detected for {no_faces}")
