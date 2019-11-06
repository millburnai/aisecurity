
"""

"aisecurity.extras.utils"

Data utils and assorted functions.

"""


import functools
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
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


class CudaEngine(object):

    # CONSTANTS
    CONSTANTS = {
        "trt_logger": trt.Logger(trt.Logger.WARNING),
        "dtype": trt.float16,
        "max_batch_size": 1,
        "max_workspace_size": 1 << 20,
    }


    # INITS
    def __init__(self, model_file, **kwargs):
        self.model_file = model_file
        self.CONSTANTS = {**self.CONSTANTS, **kwargs}

    # CUDA MANAGEMENT
    def build_cuda_engine(self, input_name, input_shape, output_name):

        with trt.Builder(self.CONSTANTS["trt_logger"]) as builder, builder.create_network() as network, \
            trt.UffParser() as parser:

            builder.max_batch_size = self.CONSTANTS["max_batch_size"]
            builder.max_workspace_size = self.CONSTANTS["max_workspace_size"]

            if self.CONSTANTS["dtype"] == trt.float16:
                builder.fp16_mode = True

            parser.register_input(input_name, input_shape)
            parser.register_output(output_name)

            parser.parse(self.model_file, network, self.CONSTANTS["dtype"])

            self.network = network
            self.config = builder.create_builder_config()

            self.engine = builder.build_cuda_engine(network)

    # MEMORY ALLOCATION
    def _malloc(self):
        # determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host i/o
        h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=self.CONSTANTS["dtype"])
        h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=self.CONSTANTS["dtype"])

        # allocate device memory for inputs and outputs
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        return h_input, d_input, h_output, d_output

    # INFERENCE
    def predict(self, img):
        h_input, d_input, h_output, d_output = self._malloc()
        np.copyto(h_input, img)

        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod(d_input, h_input)
            context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
            cuda.memcpy_dtoh(h_output, d_output)

            return h_output

    # DISPLAY
    def summary(self):
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)

            print("\nLAYER {}".format(i))
            print("===========================================")

            layer_input = layer.get_input(0)
            if layer_input:
                print("\tInput Name:  {}".format(layer_input.name))
                print("\tInput Shape: {}".format(layer_input.shape))

            layer_output = layer.get_output(0)
            if layer_output:
                print("\tOutput Name:  {}".format(layer_output.name))
                print("\tOutput Shape: {}".format(layer_output.shape))
            print("===========================================")


if __name__ == "__main__":
    print("Nothing for now!")