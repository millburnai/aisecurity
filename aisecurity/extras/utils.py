
"""

"aisecurity.extras.utils"

Data utils and assorted functions.

"""


import functools
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


    # INITS
    def __init__(self, **kwargs):
        self.CONSTANTS = {**self.CONSTANTS, **kwargs}

        self.builder = trt.Builder(CudaEngineManager.CONSTANTS["trt_logger"])
        self.builder.max_batch_size = CudaEngineManager.CONSTANTS["max_batch_size"]
        self.builder.max_workspace_size = CudaEngineManager.CONSTANTS["max_workspace_size"]

        if self.CONSTANTS["dtype"] == trt.float16:
            self.builder.fp16_mode = True

        self.network = self.builder.create_network()

    # CUDA ENGINE MANAGEMENT
    def read_cuda_engine(self, engine_file):
        with open(engine_file, "rb") as file, trt.Runtime(self.CONSTANTS["trt_logger"]) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())

    def write_cuda_engine(self, uff_file, target_file, input_name, input_shape, output_name):

        @timer("Model parsing time")
        def parse(parser, uff_file, network, input_name, input_shape, output_name):
            parser.register_input(input_name, input_shape)
            parser.register_output(output_name)

            parser.parse(uff_file, network, self.CONSTANTS["dtype"])

            return parser

        @timer("Engine building time")
        def build_engine(builder, network):
            return builder.build_cuda_engine(network)

        @timer("Engine serializing time")
        def serialize_engine(engine):
            return engine.serialize()

        self.parser = parse(trt.UffParser(), uff_file, self.network, input_name, input_shape, output_name)

        self.engine = serialize_engine(build_engine(self.builder, self.network))

        with open(target_file, "wb") as file:
            file.write(self.engine)

# FREEZE KERAS MODEL AS FROZEN GRAPH
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
        raise ValueError("{} must be a .h5 or a .pb file")

    session = tf.keras.backend.get_session()

    input_names = [layer.op.name for layer in model.inputs]
    output_names = [layer.op.name for layer in model.outputs]

    frozen_graph = _freeze_graph(session.graph, session, output_names)
    if save_dir:
        graph_io.write_graph(frozen_graph, save_dir, save_name, as_text=False)

    return frozen_graph, (input_names, output_names)



if __name__ == "__main__":
    print("Nothing for now!")
    c = CudaEngineManager()
    c.write_cuda_engine("/home/ryan/scratchpad/aisecurity/models/frozen_death.uff",
                        "/home/ryan/scratchpad/aisecurity/models/okboomer.engine",
                        "input_1", (3, 160, 160), "Bottleneck_BatchNorm/batchnorm/add_1")