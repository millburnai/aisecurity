import argparse

import uff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="frozen pb filepath")
    parser.add_argument("--outfile", help="out uff filepath")
    parser.add_argument("--frozen", help="already frozen graphdef or not",
                        action="store_true")
    args = parser.parse_args()

    with open(args.outfile, "wb+") as dump:
        if args.frozen:
            dump.write(uff.from_tensorflow_frozen_model(args.infile,
                                                        output_nodes=["embeddings"]))
        else:
             dump.write(uff.from_tensorflow_model(args.infile,
                                                  output_nodes=["embeddings"]))
