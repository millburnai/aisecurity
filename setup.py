"""

"setup.py"

Sets up aisecurity module. TensorFlow only installed if it isn't in SUPPORTED_TF_VERSIONS.

"""

from setuptools import setup, find_packages

# TENSORFLOW INSTALL
SUPPORTED_TF_VERSIONS = ["1.12", "1.14", "1.15"]

INSTALL_REQUIRES = ["adafruit-circuitpython-charlcd", "keras", "matplotlib", "mysql-connector-python", "Pyrebase",
                    "pycryptodome", "requests", "scikit-learn", "websocket-client"]

try:
    import tensorflow as tf
    assert any(version in tf.__version__ for version in SUPPORTED_TF_VERSIONS)
except (ModuleNotFoundError, AssertionError):
    INSTALL_REQUIRES.append("tensorflow==1.15.2")


# SETUP
setup(
    name="aisecurity",
    version="0.9a",
    description="CSII AI facial recognition.",
    long_description=open("README.md", encoding="utf-8").read(),
    url="https://github.com/orangese/aisecurity",
    author="Ryan Park, Liam Pilarski",
    author_email="22parkr@millburn.org, 22pilarskil@millburn.org",
    license=None,
    python_requires=">=3.5.0",
    install_requires=INSTALL_REQUIRES,
    scripts=["bin/drop.sql", "bin/make_config.sh"],
    packages=find_packages(),
    zip_safe=False
)
