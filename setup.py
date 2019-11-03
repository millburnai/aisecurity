from setuptools import setup, find_packages

setup(
    name="aisecurity",
    version="1.0a",
    description="CSII AI facial recognition.",
    long_description=open("README.md").read(),
    url="https://github.com/orangese/aisecurity",
    author="Ryan Park, Liam Pilarski",
    author_email="22parkr@millburn.org, 22pilarskil@millburn.org",
    license=None,
    python_requires=">=3.5.0",
    install_requires=["numpy", "matplotlib", "termcolor", "scikit-learn", "imageio", "mtcnn", "pycryptodome",
                      "tensorflow<=1.15.0"],
    packages=find_packages(),
    scripts=["bin/make_config.sh", "bin/make_keys.sh"],
    zip_safe=False
)
