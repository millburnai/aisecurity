
"""

"aisecurity.extras.utils"

Data and CUDA utils.

"""


import functools
import subprocess
import time

import tensorflow as tf
from tensorflow.python.framework import graph_io
import tensorrt as trt


# DECORATORS
def timer(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(time.time() - start, 3)))
            return result

        return _func

    return _timer


# CUDA ENGINE MANAGER
class CudaEngineManager(object):

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
    @staticmethod
    @timer("Uff model parsing time")
    def _parse_uff(uff_file, network, input_name, input_shape, output_name):
        parser = trt.UffParser()

        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)

        parser.parse(uff_file, network, CudaEngineManager.CONSTANTS["dtype"])

        return parser

    @staticmethod
    @timer("Caffe model parsing time")
    def _parse_caffe(caffe_model_file, caffe_deploy_file, network, output_name="prob1"):
        parser = trt.CaffeParser()

        model_tensors = parser.parse(deploy=caffe_deploy_file, model=caffe_model_file, network=network,
                                     dtype=CudaEngineManager.CONSTANTS["dtype"])

        network.mark_output(model_tensors.find(output_name))

        return parser

    @staticmethod
    @timer("Engine building time")
    def _build_engine(builder, network):
        return builder.build_cuda_engine(network)

    @staticmethod
    @timer("Engine serializing time")
    def _serialize_engine(engine):
        return engine.serialize()


    # CUDA ENGINE WRITE AND READ
    def read_cuda_engine(self, engine_file):
        with open(engine_file, "rb") as file, trt.Runtime(self.CONSTANTS["trt_logger"]) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())

    def uff_write_cuda_engine(self, uff_file, target_file, input_name, input_shape, output_name):
        self.parser = self._parse_uff(uff_file, self.network, input_name, input_shape, output_name)
        self.engine = self._serialize_engine(self._build_engine(self.builder, self.network))

        with open(target_file, "wb") as file:
            file.write(self.engine)

    def caffe_write_cuda_engine(self, caffe_model_file, caffe_deploy_file, target_file):
        self.parser = self._parse_caffe(caffe_model_file, caffe_deploy_file, self.network)
        self.engine = self._serialize_engine(self._build_engine(self.builder, self.network))

        with open(target_file, "wb") as file:
            file.write(self.engine)


# MODEL CONVERSIONS
class GraphConverter(object):

    # .h5 -> .pb
    @staticmethod
    @timer("Freezing time")
    def keras_to_frozen(path_to_model, save_dir=None, save_name="frozen_graph.pb"):

        tf.keras.backend.clear_session()

        def _freeze_graph(graph, session, output):
            with graph.as_default():
                variable = tf.graph_util.remove_training_nodes(graph.as_graph_def())
                frozen = tf.graph_util.convert_variables_to_constants(session, variable, output)
                return frozen

        tf.keras.backend.set_learning_phase(0)

        if path_to_model.endswith(".h5"):
            model = tf.keras.models.load_model(path_to_model)
        else:
            raise ValueError("{} must be a .h5 file")

        session = tf.keras.backend.get_session()

        input_names = [layer.op.name for layer in model.inputs]
        output_names = [layer.op.name for layer in model.outputs]

        frozen_graph = _freeze_graph(session.graph, session, output_names)
        if save_dir:
            graph_io.write_graph(frozen_graph, save_dir, save_name, as_text=False)

        return frozen_graph, (input_names, output_names)

    # .pb -> .uff
    @staticmethod
    @timer("Conversion to .uff time")
    def frozen_to_uff(path_to_model):
        bash_cmd = "convert-to-uff \"{}\"".format(path_to_model)
        process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error


if __name__ == "__main__":
    print("Nothing for now!")
