import argparse
import sys

sys.path.insert(1, "../")
from util.engine import CudaEngineManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="uff filepath")
    parser.add_argument("--outfile", help="out engine file")
    args = parser.parse_args()

    engine_maker = CudaEngineManager()
    engine_maker.uff_write_cuda_engine(args.infile, args.outfile,
                                       input_name="input",
                                       input_shape=(3, 160, 160),
                                       output_names=["embeddings"])
