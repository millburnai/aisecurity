from aisecurity.optim.engine import CudaEngineManager
import uff

print("yeehaw")
try: 
	uff.from_tensorflow_frozen_model("/home/aisecurity/.aisecurity/models/20180402-114759.pb", output_nodes=["embeddings"], output_filename="/home/aisecurity/.aisecurity/models/20180402-114759.uff")

	engine = CudaEngineManager()
	engine.uff_write_cuda_engine("/home/aisecurity/.aisecurity/models/20180402-114759.uff", target_file="/home/aisecurity/.aisecurity/models/20180402-114759.engine", input_name="input", input_shape=(3, 160, 160), output_name="embeddings")
except Exception as e:
	print(e)