import argparse
import os

import tensorflow.compat.v1 as tf  # noqa
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import graph_io


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="graphdef pb")
    parser.add_argument("--outfile", help="frozen graphdef pb")
    args = parser.parse_args()

    outs = ["embeddings"]

    with tf.Session() as sess:
        tf.keras.backend.set_learning_phase(0)

        with sess.graph.as_default():
            with tf.gfile.GFile(args.infile, "rb") as f:
                graphdef = tf.GraphDef()
                graphdef.ParseFromString(f.read())

            tf.import_graph_def(graphdef)
            c_graphdef = freeze_graph_def(sess, graphdef, "embeddings")

#            var = graph_util.remove_training_nodes(sess.graph.as_graph_def())
#            c_graphdef = graph_util.convert_variables_to_constants(sess, var,
#                                                                   outs)

            savedir, fname = os.path.split(args.outfile)
            graph_io.write_graph(c_graphdef, savedir, fname, as_text=False)

        import sys
        sys.exit(0)  # lmao

        nodes = []
        for node in graphdef.node:
            if node.op == "RefSwitch":
                node.op = "Switch"
                for i, node_inp in enumerate(node.input):
                    if "moving_" in node_inp:
                        node.input[i] = node_inp = "/read"
            elif node.op == "AssignSub":
                node.op = "sub"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]
            print(node.name)

        with sess.graph.as_default():
            tf.import_graph_def(graphdef)

            var = graph_util.remove_training_nodes(sess.graph.as_graph_def())
            c_graphdef = graph_util.convert_variables_to_constants(sess, var,
                                                                   outs)

            savedir, fname = os.path.split(args.outfile)
            graph_io.write_graph(c_graphdef, savedir, fname, as_text=False)
