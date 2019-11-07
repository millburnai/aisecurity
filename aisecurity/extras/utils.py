
"""

"aisecurity.extras.utils"

Data utils and assorted functions.

"""


import functools
import time

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
        "dtype": trt.float16,
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

    def write_cuda_engine(self, target_file, uff_file, input_name, input_shape, output_name):

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


if __name__ == "__main__":
    print("Nothing for now!")
    c = CudaEngineManager()
    c.write_cuda_engine("/home/ryan/scratchpad/aisecurity/models/test.engine",
                        "/home/ryan/scratchpad/aisecurity/models/ms_celeb_1m.uff",
                        "input_1", (3, 160, 160), "Bottleneck_BatchNorm/batchnorm/add_1")