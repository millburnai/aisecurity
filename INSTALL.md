# Installation instructions
This instructions will only work for `v2021.0`.

## General installation
1. Install `tensorflow>=2`, `scikit-learn`, `opencv-python`, `pycryptodome`, `websocket`, `tqdm`, `dropbox`
2. Clone this repo (branch `v2021.0`) to `aisecurity`
3. `cd aisecurity/scripts && python3 download.py <TOKEN>`, see Discord server (`announcements` channel) for `<TOKEN>`
4. `python3 facenet_test.py` - should recognize you if you are a junior

## GPU installation
Assumes CUDA, CuDNN installed.

After general installation...
1. Install `pycuda` and `tensorrt`
2. Run `make` in `util/trt_mtcnn_plugin/mtcnn`
3. Run `./create_engines` in the same directory
4. Run `make` in `util/trt_mtcnn_plugin`
5. See ["Install `facenet` engine"](#install-facenet-engine) section

## Jetson Nano installation
Follow these for Jetson-specific instructions, tested with Jetpack 4.5 (CUDA 10.2).

Make sure that wifi is connected: `sudo nmcli device wifi connect <NETWORK_NAME> password <PASSWORD>`

### Fast installation
*Only applicable if preloaded disk images are available. See the [Google Drive folder](https://drive.google.com/drive/folders/11dhxsYLuP5pNr_2hJuzqIia5U9H6yBEd?usp=sharing)*

#### Flash image
1. Download and unzip the image from the Google Drive url (use `2021a1.img.gz`)
2. Plug in the microSD card and identify the `/dev/disk[n]` path using `diskutil list`
3. Erase the microSD card using Disk Utility. Format should be changed from APFS to ExFAT
4. Write the image: `sudo dd if=[path/to/img] of=/dev/disk[n]`
   - Be VERY careful with this command- a typo will irrecoverably screw up a) the microSD card or b) your computer

#### Test everything
1. Make sure to use barrel jack PSU and have the J48 jumper cap on
2. Plug in microUSB and connect to computer
3. Find the `tty.usb[id]` file in `/dev/` and run `screen /dev/tty.usb[id]`
4. Login to the Nano; `cd aisecurity/scripts && python3 facenet_test.py`

### Regular installation
*Only applicable if `.engine`s are available.* Use this if developing on an SD card != `a1`.

#### Install dependencies
1. `sudo apt-get update`
2. `sudo apt-get install python3-pip`
3. `sudo pip3 install -U pip testresources setuptools==49.6.0`
4. `sudo pip3 install -U scikit-learn tqdm websocket Cython pycryptodome dropbox`
5. `sudo pip3 install -U numpy==1.19.4`
6. `export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}`
7. `export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

#### Install `aisecurity`
1. `cd ~/ && git clone https://github.com/orangese/aisecurity.git`
2. `cd aisecurity && git submodule update --init`
3. `cd util/trt_mtcnn_plugin && make`
4. `cd ~/aisecurity/scripts && python3 download.py <TOKEN>`, see Discord server (`announcements` channel) for `<TOKEN>`
6. `python3 facenet_test.py` to ensure that everything works

### Full installation
*Only follow these instructions if the `.engine` files aren't available yet.* 

Unless developing on the `a1` Jetson, do not use these instructions.

#### Install dependencies
1. `sudo apt-get update`
2. `sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran python3-h5py`
3. `sudo apt-get install python3-pip`
4. `sudo pip3 install -U pip testresources setuptools==49.6.0`
5. `sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11`
6. `sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow`
7. `sudo pip3 install scikit-learn`
8. `python3 -m pip install pycuda pycryptodome tqdm websocket Cython`
9. `sudo pip3 install -U numpy==1.19.4` (this step might not be necessary, just make sure that `numpy` version is `1.19.4` after step 7)
12. `export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}`
13. `export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

#### Install aisecurity and `mtcnn` models
1. `cd ~/ && git clone https://github.com/orangese/aisecurity.git`
2. `cd aisecurity && git submodule update --init`
3. `cd face/trt_mtcnn_plugin/mtcnn && make && ./create_engines`. Change engine params in `det1_relu.prototxt` or `create_engines.cpp` (before `make`) if necessary
4. `cd .. && make`
5. Change the paths in `config/config.json` so that `default_model` points to `20180402-114759.engine`.

#### Install `facenet` engine
1. `cd /usr/src/tensorrt/samples/trtexec && make`
2. `export PATH=$PATH:/usr/src/tensorrt/bin`
3. `cd ~/aisecurity/config/models && trtexec --saveEngine=20180402-114759.engine --uffNHWC --uff=20180402-114759.uff --uffInput=input,160,160,3 --fp16 --output=embeddings`
4. `cd ~/aisecurity/scripts && python3 facenet_test.py` to ensure that everything works
