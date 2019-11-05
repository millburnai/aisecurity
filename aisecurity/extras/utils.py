
"""

"aisecurity.extras.utils"

Data utils and assorted functions.

"""

import functools
import time

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io


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


# KERAS TO TF
@timer("Freezing time")
def freeze_graph(path_to_keras_model, save_dir, save_name="frozen_graph.h5"):

    tf.keras.backend.clear_session()

    def _freeze_graph(graph, session, output):
        with graph.as_default():
            variable = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            frozen = tf.graph_util.convert_variables_to_constants(session, variable, output)
            return frozen

    tf.keras.backend.set_learning_phase(0)

    model = tf.keras.models.load_model(path_to_keras_model)

    session = tf.keras.backend.get_session()

    input_names = [layer.op.name for layer in model.inputs]
    output_names = [layer.op.name for layer in model.outputs]

    frozen_graph = _freeze_graph(session.graph, session, output_names)
    graph_io.write_graph(frozen_graph, save_dir, save_name, as_text=False)

    return frozen_graph, (input_names, output_names)


# TF TO TENSORRT
@timer("Inference time")
def write_inference_graph(frozen_graph, output_names, save_dir, save_name):

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode="FP16",
        minimum_segment_size=50
    )

    graph_io.write_graph(trt_graph, save_dir, save_name, as_text=False)

    return trt_graph


if __name__ == "__main__":
    print("Nothing for now!")