"""

"aisecurity.optim.engine"

CUDA engine management.

"""

import json
import warnings

import numpy as np
from aisecurity.utils.paths import CONFIG_HOME

from aisecurity.utils.events import print_time


################################ Setup ################################

# AUTOINIT
INIT_SUCCESS = True

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except (ModuleNotFoundError, ImportError) as e:  # don't know which exception
    warnings.warn("cannot import pycuda.autoinit or pycuda.driver: '{}'".format(e))
    INIT_SUCCESS = False

try:
    import tensorrt as trt
except (ModuleNotFoundError, ImportError) as e:  # don't know which exception
    warnings.warn("cannot import tensorrt: '{}'".format(e))
    INIT_SUCCESS = False

if not INIT_SUCCESS:
    warnings.warn("tensorrt mode cannot be used: library import failed")


################################ CUDA Engine Manager ################################
class CudaEngineManager:
    """Class-oriented cuda engine management"""

    # CONSTANTS
    CONSTANTS = {
        "logger": None,
        "dtype": None,
        "max_batch_size": 1,
        "max_workspace_size": 1 << 20,
    }


    # INITS
    def __init__(self, **kwargs):
        """Initializes CudaEngineManager

        :param kwargs: overrides CudaEngineManager.CONSTANTS

        """

        # constants (have to be set here in case trt isn't imported)
        self.CONSTANTS["logger"] = trt.Logger(trt.Logger.ERROR)
        self.CONSTANTS["dtype"] = trt.float32

        self.CONSTANTS = {**self.CONSTANTS, **kwargs}

        # builder and netork
        self.builder = trt.Builder(CudaEngineManager.CONSTANTS["logger"])
        self.builder.max_batch_size = CudaEngineManager.CONSTANTS["max_batch_size"]
        self.builder.max_workspace_size = CudaEngineManager.CONSTANTS["max_workspace_size"]

        if self.CONSTANTS["dtype"] == trt.float16:
            self.builder.fp16_mode = True

        self.network = self.builder.create_network()


    # MEMORY ALLOCATION
    def allocate_buffers(self):
        """Allocates GPU memory for future use and creates an asynchronous stream"""

        # determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host i/o
        self.h_input = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(0)),
            dtype=trt.nptype(self.CONSTANTS["dtype"])
        )
        self.h_output = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(1)),
            dtype=trt.nptype(self.CONSTANTS["dtype"])
        )

        # allocate device memory for inputs and outputs
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()


    # INFERENCE
    def inference(self, img, output_shape=None):
        """Run inference on given image

        :param img: input image array
        :param output_shape: shape to which to reshape the output array (default: None)
        :returns: output array

        """

        def buffer_ready(arr):
            arr = arr.astype(trt.nptype(CudaEngineManager.CONSTANTS["dtype"]))
            arr = arr.transpose(0, 3, 1, 2).ravel()
            return arr

        np.copyto(self.h_input, buffer_ready(img))

        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            context.execute_async(
                batch_size=1,
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

        output = np.copy(self.h_output)
        if output_shape is not None:
            output = output.reshape(output_shape)

        return output


    # CUDA ENGINE READ/WRITE
    def read_cuda_engine(self, engine_file):
        """Read and deserialize engine from file

        :param engine_file: path to engine file

        """

        with open(engine_file, "rb") as file, trt.Runtime(self.CONSTANTS["logger"]) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())

    @print_time("Engine building and serializing time")
    def build_and_serialize_engine(self):
        """Builds and serializes a cuda engine"""
        self.engine = self.builder.build_cuda_engine(self.network).serialize()

    @print_time("uff model parsing time")
    def parse_uff(self, uff_file, input_name, input_shape, output_name):
        """Parses .uff file and prepares for serialization

        :param uff_file: path to uff model
        :param input_name: name of input
        :param input_shape: input shape (channels first)
        :param output_name: name of output

        """

        parser = trt.UffParser()

        # input shape must always be channels-first
        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)

        parser.parse(uff_file, self.network, CudaEngineManager.CONSTANTS["dtype"])

        self.parser = parser

    @print_time("caffe model parsing time")
    def parse_caffe(self, caffe_model_file, caffe_deploy_file, output_name="prob1"):
        """Parses caffe model file and prepares for serialization

        :param caffe_model_file: path to caffe model file
        :param caffe_deploy_file: path to caffe deploy file
        :param output_name: output name

        """

        parser = trt.CaffeParser()

        model_tensors = parser.parse(
            deploy=caffe_deploy_file, model=caffe_model_file, network=self.network,
            dtype=CudaEngineManager.CONSTANTS["dtype"]
        )

        self.network.mark_output(model_tensors.find(output_name))

        self.parser = parser

    def uff_write_cuda_engine(self, uff_file, target_file, input_name, input_shape, output_name):
        """Parses a uff model and writes it as a serialized cuda engine

        :param uff_file: uff filepath
        :param target_file: target filepath for engine
        :param input_name: name of input
        :param input_shape: input shape (channels first)
        :param output_name: name of output

        """

        self.parse_uff(uff_file, input_name, input_shape, output_name)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)

    def caffe_write_cuda_engine(self, caffe_model_file, caffe_deploy_file, output_name, target_file):
        """Parses a caffe model and writes it as a serialized cuda engine

        :param caffe_model_file: path to caffe model
        :param caffe_deploy_file: path to caffe deploy file
        :param output_name: name of output
        :param target_file: target filepath for engine

        """

        self.parse_caffe(caffe_model_file, caffe_deploy_file, output_name)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)


    # DISPLAY
    def summary(self):
        """Printed summary of all the layers of the network"""
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


################################ CUDA Engine ################################
class CudaEngine:
    """Class-oriented cuda engine manager wrapper"""

    # PREBUILT MODELS
    MODELS = json.load(open(CONFIG_HOME + "/config/cuda_models.json"))


    # INITS
    def __init__(self, filepath, input_name, output_name, input_shape, **kwargs):
        """Initializes a cuda engine

        :param filepath: path to engine file
        :param input_name: name of input
        :param output_name: name of output
        :param input_shape: input shape (channels first)
        :param kwargs: overrides CudaEngineManager.CONSTANTS

        """

        # engine
        self.engine_manager = CudaEngineManager(**kwargs)
        self.engine_manager.read_cuda_engine(filepath)

        # input and output shapes and names
        self.io_check(filepath, input_name, output_name, input_shape)

        # memory allocation
        self.engine_manager.allocate_buffers()

    def io_check(self, filepath, input_name, output_name, input_shape):
        """Checks that I/O names and shapes are provided or detected

        :param filepath: path to engine file
        :param input_name: provided name of input
        :param output_name: provided name of output
        :param input_shape: provided input shape
        :raises: AssertionError: if I/O name and shape is not detected or provided

        """

        self.input_name, self.output_name, self.model_name = None, None, None

        for model in self.MODELS:
            if model in filepath:
                self.model_name = model
                self.input_name = self.MODELS[model]["input"]
                self.output_name = self.MODELS[model]["output"]

        if input_name:
            self.input_name = input_name
        if output_name:
            self.output_name = output_name

        if input_shape:
            assert input_shape[0] == 3, "input shape to engine should be in channels-first mode"
            self.input_shape = input_shape
        elif self.model_name is not None:
            self.input_shape = self.MODELS[self.model_name]["input_shape"]

        assert self.input_name and self.output_name, "I/O names for {} not detected or provided".format(filepath)
        assert self.input_shape, "input shape for {} not detected or provided".format(filepath)


    # INFERENCE
    def inference(self, *args, **kwargs):
        """Inference on given image

        :param args: args to CudaEngineManager().inference()
        :param kwargs: kwargs to CudaEngineManager().inference()

        """

        return self.engine_manager.inference(*args, **kwargs)


    # SUMMARY
    def summary(self):
        """Printed summary of all engine layers"""
        self.engine_manager.summary()
