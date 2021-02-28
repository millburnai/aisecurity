# Installation instructions
This instructions will only work for `v2021.0`.

## General installation
1. Install `tensorflow>=2`, `scikit-learn`, `opencv-python`, `pycryptodome`, `websocket`, `tqdm`
2. Clone this repo (branch `v2021.0`) to `aisecurity`
3. Download Dropbox `config` folder in `2020-2021` (don't have access? ask!)
4. Move `config` into `aisecurity/config`
5. Change paths in `config/config.json` so that they point to the right files

## GPU installation
Assumes CUDA, CuDNN installed (or if on Jetson).

After general installation...
1. Install `pycuda` and `tensorrt`
2. Run `make` in `face/trt_mtcnn_plugin/mtcnn`
3. Run `./create_engines` in the same directory
4. Run `make` in `face/trt_mtcnn_plugin`
5. Run `deployment/make_engines.py` with the correct args
