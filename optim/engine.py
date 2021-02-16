"""PyCUDA + TensorRT engine management.
"""

import json
import sys

import numpy as np

sys.path.insert(1, "../")
from utils.paths import config_home  # noqa


try:
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda  # noqa
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] cannot import pycuda.autoinit or pycuda.driver: '{e}'")

try:
    import tensorrt as trt  # noqa
except (ModuleNotFoundError, ImportError) as e:
    print(f"[DEBUG] cannot import tensorrt: '{e}'")


class CudaEngineManager:
    """Cuda engine management and interface with GPU using pycuda, trt"""

    # INITS
    def __init__(self, logger=None, dtype=None, max_batch_size=1,
                 max_workspace_size=1 << 20):
        """Initializes CudaEngineManager
        :param logger: trt logger (default: trt.Logger(trt.Logger.ERROR))
        :param dtype: trt dtype (default: trt.float32)
        :param max_batch_size: max batch size (default: 1)
        :param max_workspace_size: max workspace size (default: 1 << 20)
        """

        self.logger = logger if logger else trt.Logger(trt.Logger.ERROR)
        self.dtype = dtype if dtype else trt.float32
        self.max_workspace_size = max_workspace_size

        # builder and netork
        self.builder = trt.Builder(self.logger)
        self.builder.max_batch_size = max_batch_size
        self.builder.max_workspace_size = max_workspace_size

        if self.dtype == trt.float16:
            self.builder.fp16_mode = True

        self.network = self.builder.create_network()

    def allocate_buffers(self):
        """Allocates GPU memory for future use and creates an asynchronous stream"""
        self.h_input = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(self.dtype)
        )
        self.h_output = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(self.dtype)
        )
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()

    def create_context(self):
        """Creates execution context for engine"""
        self.context = self.engine.create_execution_context()

    def inference(self, imgs):
        """Run inference on given images
        :param imgs: input image arrays
        :returns: output array
        """

        def buffer_ready(arr, dtype):
            arr = arr.astype(dtype)
            arr = arr.transpose(0, 3, 1, 2).ravel()
            return arr

        outputs = np.empty((len(imgs), *self.h_output.shape))
        for idx, img in enumerate(np.expand_dims(imgs, axis=1)):
            np.copyto(self.h_input, buffer_ready(img, trt.nptype(self.dtype)))

            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async(
                batch_size=1, bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            np.copyto(outputs[idx], self.h_output)

        return outputs

    def read_cuda_engine(self, engine_file):
        """Read and deserialize engine from file
        :param engine_file: path to engine file
        """

        with open(engine_file, "rb") as file, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())

    def build_and_serialize_engine(self):
        """Builds and serializes a cuda engine"""
        self.engine = self.builder.build_cuda_engine(self.network).serialize()

    def parse_uff(self, uff_file, input_name, input_shape, output_names):
        """Parses .uff file and prepares for serialization
        :param uff_file: path to uff model
        :param input_name: name of input
        :param input_shape: input shape (channels first)
        :param output_names: names of outputs
        """

        parser = trt.UffParser()

        # input shape must always be channels-first
        parser.register_input(input_name, input_shape)

        for output in output_names:
            parser.register_output(output)

        parser.parse(uff_file, self.network, self.dtype)

        self.parser = parser

    def parse_caffe(self, caffe_model_file, caffe_deploy_file, output_names):
        """Parses caffe model file and prepares for serialization
        :param caffe_model_file: path to caffe model file
        :param caffe_deploy_file: path to caffe deploy file
        :param output_names: output names
        """

        parser = trt.CaffeParser()

        model_tensors = parser.parse(
            deploy=caffe_deploy_file, model=caffe_model_file,
            network=self.network, dtype=self.dtype
        )

        for output in output_names:
            self.network.mark_output(model_tensors.find(output))

        self.parser = parser

    def uff_write_cuda_engine(self, uff_file, target_file, input_name,
                              input_shape, output_names):
        """Parses a uff model and writes it as a serialized cuda engine
        :param uff_file: uff filepath
        :param target_file: target filepath for engine
        :param input_name: name of input
        :param input_shape: input shape (channels first)
        :param output_names: name of outputs
        """

        self.parse_uff(uff_file, input_name, input_shape, output_names)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)

    def caffe_write_cuda_engine(self, caffe_model_file, caffe_deploy_file,
                                output_names, target_file):
        """Parses a caffe model and writes it as a serialized cuda engine
        :param caffe_model_file: path to caffe model
        :param caffe_deploy_file: path to caffe deploy file
        :param output_names: name of outputs
        :param target_file: target filepath for engine
        """

        self.parse_caffe(caffe_model_file, caffe_deploy_file, output_names)
        self.build_and_serialize_engine()

        with open(target_file, "wb") as file:
            file.write(self.engine)


class CudaEngine:
    """Cuda engine manager wrapper for interfacing with FaceNet class"""
    MODELS = json.load(open(config_home + "/defaults/cuda_models.json",
                            encoding="utf-8"))

    def __init__(self, filepath, input_name, output_name, input_shape,
                 **kwargs):
        """Initializes a cuda engine
        :param filepath: path to engine file
        :param input_name: name of input
        :param output_name: name of output
        :param input_shape: input shape (channels first)
        :param kwargs: overrides CudaEngineManager settings
        """

        # engine
        self.engine_manager = CudaEngineManager(**kwargs)
        self.engine_manager.read_cuda_engine(filepath)

        # input and output shapes and names
        self.io_check(filepath, input_name, output_name, input_shape)

        # memory allocation
        self.engine_manager.allocate_buffers()
        self.engine_manager.create_context()

    def io_check(self, filepath, input_name, output_name, input_shape):
        """Checks that I/O names and shapes are provided or detected
        :param filepath: path to engine file
        :param input_name: provided name of input
        :param output_name: provided name of output
        :param input_shape: provided input shape
        :raises: AssertionError: if I/O name and shape is not detected/provided
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
            assert input_shape[0] == 3, \
                "input shape to engine should be in channels-first mode"
            self.input_shape = input_shape
        elif self.model_name is not None:
            self.input_shape = self.MODELS[self.model_name]["input_shape"]

        assert self.input_name and self.output_name, \
            f"I/O names for {filepath} not detected or provided"
        assert self.input_shape, \
            f"input shape for {filepath} not detected or provided"


    # INFERENCE
    def inference(self, *args, **kwargs):
        """Inference on given image
        :param args: args to CudaEngineManager().inference()
        :param kwargs: kwargs to CudaEngineManager().inference()
        """

        return self.engine_manager.inference(*args, **kwargs)
