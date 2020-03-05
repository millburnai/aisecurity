# Installation of `aisecurity`

## Prerequisites

1. Python >= 3.5

2. `curl` must be installed. It should be pre-installed with Mac systems and can be installed using `sudo apt install curl` on Linux systems. For Windows users, go to https://stackoverflow.com/questions/9507353/how-do-i-install-and-use-curl-on-windows and follow those instructions.

3. For GPU usage, `CUDA>=9.0` and the appropriate version of `CuDNN` must be installed (ignore if part of AI Security team).

5. Only TensorFlow versions 1.12.x, 1.14.x, 1.15.x are supported. Tensorflow 2.0 is not yet supported.

6. To use database logging, install necessary prerequisite packages (not need for developers).

## Installation

> `python3 -m pip install "git+https://github.com/orangese/aisecurity.git@v0.9a"`

After installing, you might want to change the key location settings in `~/.aisecurity/aisecurity.json`, which is installed with the `aisecurity` package after the first import.

## Upgrade

> `python3 -m pip install --upgrade "git+https://github.com/orangese/aisecurity.git@v0.9a"`

## FaceNet weight files

All weight files (.h5 and .pb models) are available in this [Dropbox folder](https://www.dropbox.com/sh/k9ci2nphj7i7dde/AACaQuxUJ6GoPHFxW6FtJlZca?dl=0).

