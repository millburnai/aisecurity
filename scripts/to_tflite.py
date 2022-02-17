import argparse
import tensorflow as tf


def shape(x):
    try:
        return tuple(map(int, x.split(",")))
    except:
        raise argparse.ArgumentTypeError("input must be shape tuple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input filename")
    parser.add_argument("--outfile", help="output filename")
    parser.add_argument("--inputs", help="input names", nargs="+")
    parser.add_argument("--outputs", help="output names", nargs="+")
    parser.add_argument("--input_shapes", help="input shapes", nargs="+", type=shape)
    args = parser.parse_args()

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        args.infile,
        args.inputs,
        args.outputs,
        {
            input_name: shape
            for input_name, shape in zip(args.inputs, args.input_shapes)
        },
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(args.outfile, "wb") as f:
        f.write(tflite_model)
