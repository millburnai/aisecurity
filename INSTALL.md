# Installation of `aisecurity`

## Prerequisites

1. `wget` must be installed, preferably using `brew` (install `brew` with `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`). `wget` can be installed using `brew install wget` and is pre-installed on Linux systems.

2. For GPU usage, `CUDA==9.0` and `CuDNN==7.4.1` must be installed.

3. MySQL Community Server must be downloaded (see https://dev.mysql.com/downloads/mysql/).

## Installation

If you're using MacOS Catalina (and the `zsh` shell), you must put quotations around the installation link
in order for it to work.

CPU: `pip3 install git+https://github.com/orangese/aisecurity/tree/keras`

GPU: `pip3 install git+https://github.com/orangese/aisecurity.git`

After installing, you might want to change the key location settings in `config.json`, which is installed with the `aisecurity` package.

## Upgrade

CPU: `pip3 install --upgrade git+https://github.com/orangese/tree/keras`

GPU: `pip3 install --upgrade git+https://github.com/orangese/aisecurity.git`

## Development

Because the models are stored on Git LFS, cloning or pulling might overwrite the h5 files such that they do not contain a valid `keras` model anymore. Any developments might need to re-download the correct weight files from [this Dropbox folder](https://www.dropbox.com/sh/k9ci2nphj7i7dde/AACaQuxUJ6GoPHFxW6FtJlZca?dl=0).
