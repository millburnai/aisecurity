import tensorflow as tf
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    "/Users/ethan/aisecurity/config/models/20180402-114759_opt.pb",
    ["input"],
    ["embeddings"],
    {"input":(1, 160, 160, 3)},
    )
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
