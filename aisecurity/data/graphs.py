"""

"aisecurity.data.graphs"

Data and CUDA utils.

"""


import functools
import subprocess
import time

import tensorflow as tf
from tensorflow.python.framework import graph_io
import tensorrt as trt

from aisecurity.utils.events import timer


# CUDA ENGINE MANAGER
class CudaEngineManager:

    # CONSTANTS
    CONSTANTS = {
        "trt_logger": trt.Logger(trt.Logger.WARNING),
        "dtype": trt.float32,
        "max_batch_size": 1,
        "max_workspace_size": 1 << 20,
    }


    # INIT
    def __init__(self, **kwargs):
        self.CONSTANTS = {**self.CONSTANTS, **kwargs}

        self.builder = trt.Builder(CudaEngineManager.CONSTANTS["trt_logger"])
        self.builder.max_batch_size = CudaEngineManager.CONSTANTS["max_batch_size"]
        self.builder.max_workspace_size = CudaEngineManager.CONSTANTS["max_workspace_size"]

        if self.CONSTANTS["dtype"] == trt.float16:
            self.builder.fp16_mode = True

        self.network = self.builder.create_network()


    # ENGINE TASK HELPERS
    @timer("Uff model parsing time")
    def _parse_uff(uff_file, network, input_name, input_shape, output_name):
        parser = trt.UffParser()

        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)

        parser.parse(uff_file, network, CudaEngineManager.CONSTANTS["dtype"])

        self.parser = parser

    @timer("Caffe model parsing time")
    def _parse_caffe(self, caffe_model_file, caffe_deploy_file, network, output_name="prob1"):
        parser = trt.CaffeParser()

        model_tensors = parser.parse(deploy=caffe_deploy_file, model=caffe_model_file, network=network,
                                     dtype=CudaEngineManager.CONSTANTS["dtype"])

        network.mark_output(model_tensors.find(output_name))

        self.parser = parser

    @timer("Engine building time")
    def _build_engine(self, builder, network):
        self.builder = builder.build_cuda_engine(network)

    @timer("Engine serializing time")
    def _serialize_engine(self, engine):
        self.engine = engine.serialize()


    # CUDA ENGINE WRITE AND READ
    def read_cuda_engine(self, engine_file):
        with open(engine_file, "rb") as file, trt.Runtime(self.CONSTANTS["trt_logger"]) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())

    def uff_write_cuda_engine(self, uff_file, target_file, input_name, input_shape, output_name):
        self._parse_uff(uff_file, self.network, input_name, input_shape, output_name)
        self._build_engine(self.builder, self.network)
        self._serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)

    def caffe_write_cuda_engine(self, caffe_model_file, caffe_deploy_file, target_file):
        self.parser = self._parse_caffe(caffe_model_file, caffe_deploy_file, self.network)
        self.engine = self._serialize_engine(self._build_engine(self.builder, self.network))

        with open(target_file, "wb") as file:
            file.write(self.engine)


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
