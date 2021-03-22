import argparse
import sys

sys.path.insert(1, "../")
from util.engine import CudaEngineManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="uff filepath")
    parser.add_argument("--outfile", help="out engine file")
    parser.add_argument("--fp16", help="fp16 or not", action="store_true")
    args = parser.parse_args()

    engine_maker = CudaEngineManager(fp16=args.fp16)
    if args.infile.endswith(".uff"):
        engine_maker.uff_write_cuda_engine(args.infile, args.outfile,
                                           input_name="input",
                                           input_shape=(3, 160, 160),
                                           output_names=["embeddings"])
    elif args.infile.endswith(".onnx"):
        engine_maker.onnx_write_cuda_engine(args.infile, args.outfile)
    else:
        raise TypeError(f"{args.infile} not supported")
    print(f"[DEBUG] wrote engine to {args.outfile}")
