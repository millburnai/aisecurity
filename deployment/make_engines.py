import argparse
import sys

sys.path.insert(1, "../")
from optim.engine import CudaEngineManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uff_file", help="uff filepath")
    parser.add_argument("--target_file", help="out engine file")
    args = parser.parse_args()

    engine_maker = CudaEngineManager()
    engine_maker.uff_write_cuda_engine(args.uff_file, args.target_file,
                                       input_name="input",
                                       input_shape=(3, 160, 160),
                                       output_names=["embeddings"])
