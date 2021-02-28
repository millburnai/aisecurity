import argparse

import uff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="frozen pb filepath")
    parser.add_argument("--outfile", help="out uff filepath")
    args = parser.parse_args()

    with open(args.outfile, "wb+") as dump:
        dump.write(uff.from_tensorflow_frozen_model(args.infile,
                                                    output_nodes=["embeddings"]))
