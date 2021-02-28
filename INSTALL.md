# Installation instructions
This instructions will only work for `v2021.0`.

## General installation
1. Install `tensorflow>=2`, `scikit-learn`, `opencv-python`, `pycryptodome`, `websocket`, `tqdm`
2. Clone this repo (branch `v2021.0`) to `aisecurity`
3. Download Dropbox `config` folder in `2020-2021` (don't have access? ask!)
4. Move `config` into `aisecurity/config`
5. Change paths in `config/config.json` so that they point to the right files

## GPU installation
Assumes CUDA, CuDNN installed.

After general installation...
1. Install `pycuda` and `tensorrt`
2. Run `make` in `face/trt_mtcnn_plugin/mtcnn`
3. Run `./create_engines` in the same directory
4. Run `make` in `face/trt_mtcnn_plugin`
5. Run `deployment/make_engines.py` with the correct args

## Jetson Nano installation
Follow these for Jetson-specific instructions, tested with Jetpack 4.5 (CUDA 10.2).

### Regular installation
**Only applicable if `.engine`s are available.**

#### Install dependencies
1. `sudo apt-get update`
2. `sudo apt-get install python3-pip`
3. `sudo pip3 install -U pip testresources setuptools==49.6.0`
4. `export PATH=/usr/local/cuda/bin:$PATH ; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
5. `sudo pip3 install -U opencv-python scikit-learn tqdm websocket Cython pycryptodome numpy==1.19.4`

#### Install `aisecurity`
1. `cd ~/ ; git clone https://github.com/orangese/aisecurity.git`
2. `cd aisecurity ; git submodule update --init`
3. `cd face/trt_mtcnn_plugin ; make`
4. Download the database/model/keys/config from `2020-2021/config`. Make sure the `engine` files are present.
5. Move the `det{n}.engine` files to `face/trt_mtcnn_plugin/mtcnn/`.
6. Change the paths in `config/config.json` so that the prefixes (`/home/../aisecurity`) are correct, and so that the `default_model` points to `20180402-114759.engine` rather than `20180402-114759_opt.pb`. Change database and key paths as necessary as well.
7. Edit `facenet_test.py` to use `detector="trt-mtcnn"` and then run `python3 facenet_test.py` to ensure that everything works.

### Full installation
**Only follow these instructions if the `.engine` files aren't available yet`.**

#### Install dependencies
1. `sudo apt-get update`
2. `sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran python3-h5py`
3. `sudo apt-get install python3-pip`
4. `sudo pip3 install -U pip testresources setuptools==49.6.0`
5. `sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11`
6. `sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow`
7. `sudo pip3 install scikit-learn`
8. `export PATH=/usr/local/cuda/bin:$PATH ; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
9. `python3 -m pip install pycuda pycryptodome tqdm websocket opencv-python Cython`
10. `sudo pip3 install -U numpy==1.19.4` (this step might not be necessary, just make sure that `numpy` version is `1.19.4` after step 8)

#### Install aisecurity and `mtcnn` models
1. `cd ~/ ; git clone https://github.com/orangese/aisecurity.git`
2. `cd aisecurity ; git submodule update --init`
3. `cd face/trt_mtcnn_plugin/mtcnn ; make ; ./create_engines`
4. `cd .. ; make`
5. Download `2020-2021/config` from the Dropbox (either manually or using `deployment/download.sh`)
6. Change the paths in `config/config.json` so that the prefixes (`/home/../aisecurity`) are correct, and so that the `default_model` points to `20180402-114759.engine` rather than `20180402-114759_opt.pb`. Change database and key paths as necessary as well.

#### Install `facenet` engine
1. Run `import uff`. You should see an error saying `tensorflow has no attribute 'GraphDef'` or something like that. `sudo nano` or `sudo vim` in the path provided in the stack trace, and replace `import tensorflow as tf` with `import tensorflow.compat.v1 as tf` and `from tensorflow import GraphDef` with `from tensorflow.compat.v1 import GraphDef`. Edit all of these `tensorflow` import lines EXCEPT FOR those `import`ing `Gfile`.
2. Repeat step 2 until you can `import uff` without any problems.
3. `cd ~/aisecurity/deployment ; python3 convert_to_uff.py --infile ../config/models/20180402-114759.pb --outfile ../config/models/20180402-114759.uff`
4. `python3 make_engines.py --uff_file ../config/models/20180402-114759.uff --target_file ../config/models/20180402-114759.engine`
5. Edit `facenet_test.py` to use `detector="trt-mtcnn"` and then run `python3 facenet_test.py` to ensure that everything works.
