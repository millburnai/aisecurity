from setuptools import setup, find_packages

setup(
    name="aisecurity",
    version="0.0a",
    description="CSII AI facial recognition.",
    long_description=open("README.md").read(),
    url="https://github.com/orangese/aisecurity",
    author="Ryan Park",
    author_email="22parkr@millburn.org",
    license=None,
    python_requires=">=3.5.0",
    install_requires=["numpy", "matplotlib", "termcolor", "scikit-learn", "imageio", "mtcnn", "pycryptodome"],
    packages=find_packages(),
    scripts=["bin/make_config.sh", "bin/make_keys.sh"],
    zip_safe=False
)
