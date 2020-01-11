"""

"aisecurity.optim.engine"

CUDA engine management.

"""

import warnings

import numpy as np

from aisecurity.utils.events import timer


# AUTOINIT
INIT_SUCCESS = True

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except Exception as e:  # don't know which exception
    warnings.warn("cannot import pycuda.autoinit or pycuda.driver: '{}'".format(e))
    INIT_SUCCESS = False

try:
    import tensorrt as trt
except Exception as e:  # don't know which exception
    warnings.warn("cannot import tensorrt: '{}'".format(e))
    INIT_SUCCESS = False

if not INIT_SUCCESS:
    warnings.warn("tensorrt mode cannot be used: library import failed")


# CUDA ENGINE MANAGER
class CudaEngineManager:

    # CONSTANTS
    CONSTANTS = {
        "trt_logger": None,
        "dtype": None,
        "max_batch_size": 1,
        "max_workspace_size": 1 << 20,
    }

    # PREBUILT MODELS
    MODELS = {
        "ms_celeb_1m": {
            "input": "input_1",
            "output": "Bottleneck_BatchNorm/batchnorm/add_1",
            "input_shape": (3, 160, 160)
        },
        "vgg_face_2": {
            "input": "base_input",
            "output": "classifier_low_dim/Softmax",
            "input_shape": (3, 224, 224)
        },
        "20180402-114759": {
            "input": "batch_join",
            "output": "embeddings",
            "input_shape": (3, 160, 160)
        }
    }


    # INITS
    def __init__(self, filepath, input_name, output_name, input_shape, **kwargs):
        # constants (have to be set here in case trt isn't imported)
        self.CONSTANTS["trt_logger"] = trt.Logger(trt.Logger.WARNING)
        self.CONSTANTS["dtype"] = trt.float32

        self.CONSTANTS = {**self.CONSTANTS, **kwargs}

        # builder and netork
        self.builder = trt.Builder(CudaEngineManager.CONSTANTS["trt_logger"])
        self.builder.max_batch_size = CudaEngineManager.CONSTANTS["max_batch_size"]
        self.builder.max_workspace_size = CudaEngineManager.CONSTANTS["max_workspace_size"]

        if self.CONSTANTS["dtype"] == trt.float16:
            self.builder.fp16_mode = True

        self.network = self.builder.create_network()

        # engine
        self.read_cuda_engine(filepath)

        # input and output shapes and names
        self._io_init(filepath, input_name, output_name, input_shape)

        # memory allocation
        self.allocate_buffers()

    def _io_init(self, filepath, input_name, output_name, input_shape):
        self.input_name, self.output_name, self.model_name = None, None, None

        for model in self.MODELS:
            if model in filepath:
                self.model_name = model
                self.input_name = self.MODELS[model]["input"]
                self.output_name = self.MODELS[model]["output"]

        self.input_name, self.output_name = input_name, output_name

        if input_shape:
            assert input_shape[0] == 3, "input shape to engine should be in channels-first mode"
            self.input_shape = input_shape
        elif self.model_name is not None:
            self.input_shape = self.MODELS[self.model_name]["input_shape"]

        assert self.input_name and self.output_name, "I/O for {} not detected or provided".format(filepath)
        assert self.input_shape, "input shape for {} not detected or provided".format(filepath)


    # MEMORY ALLOCATION
    def allocate_buffers(self):
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

        output = self.h_output
        if output_shape is not None:
            output = output.reshape(output_shape)

        return output


    # CUDA ENGINE READ
    def read_cuda_engine(self, engine_file):
        with open(engine_file, "rb") as file, trt.Runtime(self.CONSTANTS["trt_logger"]) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())


    # CUDA ENGINE WRITE
    @timer("Engine building and serializing time")
    def build_and_serialize_engine(self):
        self.engine = self.builder.build_cuda_engine(self.network).serialize()

    @timer("uff model parsing time")
    def parse_uff(self, uff_file, input_name, input_shape, output_name):
        parser = trt.UffParser()

        # input shape must always be channels-first
        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)

        parser.parse(uff_file, self.network, CudaEngineManager.CONSTANTS["dtype"])

        self.parser = parser

    @timer("caffe model parsing time")
    def parse_caffe(self, caffe_model_file, caffe_deploy_file, output_name="prob1"):
        parser = trt.CaffeParser()

        model_tensors = parser.parse(
            deploy=caffe_deploy_file, model=caffe_model_file, network=self.network,
            dtype=CudaEngineManager.CONSTANTS["dtype"]
        )

        self.network.mark_output(model_tensors.find(output_name))

        self.parser = parser

    def uff_write_cuda_engine(self, uff_file, target_file, input_name, input_shape, output_name):
        self.parse_uff(uff_file, input_name, input_shape, output_name)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)

    def caffe_write_cuda_engine(self, caffe_model_file, caffe_deploy_file, output_name, target_file):
        self.parse_caffe(caffe_model_file, caffe_deploy_file, output_name)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)


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
