# Installation instructions
This instructions will only work for `v2021.0`.

## General installation
0. Install `tensorflow>=2`, `scikit-learn`, `numpy`, `opencv-python`, `pycryptodome`, `websocket`, `tqdm`
1. Clone this repo (branch `v2021.0`) to `aisecurity`
2. Download Dropbox `config` folder in `2020-2021` (don't have access? ask!)
3. Move `config` into `aisecurity/config`
4. Change paths in `config/config.json` so that they point to the right files

## GPU installation
Assumes CUDA, CuDNN installed (or if on Jetson).

After general installation...
5. Run `make` in `face/trt_mtcnn_plugin/mtcnn`
6. Run `./create_engines` in the same directory
7. Run `make` in `face/trt_mtcnn_plugin`
8. Run `deployment/make_engines.py` with the correct args
