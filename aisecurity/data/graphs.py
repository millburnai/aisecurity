"""

"aisecurity.data.graphs"

Graph control and flow.

"""

import subprocess

from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_io

from aisecurity.utils.events import timer


# MODEL CONVERSIONS

# HELPERS
def _freeze_graph(graph, sess, output_names, save_dir=None, save_name=None):
    def freeze(graph, sess, output):
        with graph.as_default():
            variable = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            frozen = tf.graph_util.convert_variables_to_constants(sess, variable, output)
            return frozen

    frozen_graph = freeze(graph, sess, output_names)

    if save_dir:
        graph_io.write_graph(frozen_graph, save_dir, save_name, as_text=False)

    return frozen_graph


# .pb -> frozen .pb
@timer("Freeze TF model time")
def freeze_graph(path_to_model, output_names, save_dir=None, save_name="frozen_graph.pb"):
    assert path_to_model.endswith(".pb"), "{} must be a .pb file".format(path_to_model)

    K.clear_session()
    K.set_learning_phase(0)

    sess = K.get_session()

    with tf.Session(graph=tf.Graph()) as sess:

        with tf.gfile.FastGFile(path_to_model, "rb") as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())

        tf.import_graph_def(graph_def, name="")

        frozen_graph = _freeze_graph(sess.graph, sess, output_names, save_dir, save_name)

        return frozen_graph


# .h5 -> frozen .pb
@timer("Freeze Keras model time")
def freeze_keras_model(path_to_model, save_dir=None, save_name="frozen_graph.pb"):
    assert path_to_model.endswith(".h5"), "{} must be a .h5 file".format(path_to_model)

    K.clear_session()
    K.set_learning_phase(0)

    model = tf.keras.models.load_model(path_to_model)

    input_names = [layer.op.name for layer in model.inputs]
    output_names = [layer.op.name for layer in model.outputs]

    sess = K.get_session()

    frozen_graph = _freeze_graph(sess.graph, sess, output_names, save_dir, save_name)

    return frozen_graph, (input_names, output_names)


# .pb -> .uff
@timer("Conversion to .uff time")
def frozen_to_uff(path_to_model):
    bash_cmd = "convert-to-uff \"{}\"".format(path_to_model)
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error
