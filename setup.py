from setuptools import setup, find_packages

# TENSORFLOW INSTALL
SUPPORTED_TF_VERSIONS = ["1.12", "1.14", "1.15"]

install_requires = ["adafruit-circuitpython-charlcd", "keras", "matplotlib",
                    "mysql-connector-python", "Pyrebase", "pycryptodome", "requests", "scikit-learn", "mtcnn"]

try:
    import tensorflow as tf
    assert any(version in tf.__version__ for version in SUPPORTED_TF_VERSIONS)
except (ModuleNotFoundError, AssertionError):
    install_requires.append("tensorflow==1.15.2")


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
    install_requires=install_requires,
    scripts=["bin/drop.sql", "bin/make_config.sh", "bin/make_keys.sh"],
    packages=find_packages(),
    zip_safe=False
)
