# aisecurity

AI-driven security system built by CSII AI for Millburn High School.

## Authors

CSII AI head: Ryan Park

CSII AI VP: Liam Pilarski

## Usage

Note that for GPU usage, `CUDA==9.0` and `CuDNN==7.4.1` must be installed.

Additionally, if you're using MacOS Catalina (and the `zsh` shell), then you must put quotations around the installation link
in order for it to work. 

Ex: `pip3 install "git+git+https://github.com/orangese/aisecurity.git#egg=aisecurity[cpu]"`

### Installation

CPU: `pip3 install git+https://github.com/orangese/aisecurity.git#egg=aisecurity[cpu]`

GPU: `pip3 install git+https://github.com/orangese/aisecurity.git#egg=aisecurity[gpu]`

### Upgrade

CPU: `pip3 install --upgrade git+https://github.com/orangese/aisecurity.git#egg=aisecurity[cpu]`

GPU: `pip3 install --upgrade git+https://github.com/orangese/aisecurity.git#egg=aisecurity[gpu]`
