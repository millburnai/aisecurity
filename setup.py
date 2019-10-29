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
    python_requires=">=3.6.0",
    install_requires=["numpy", "scipy", "keras", "matplotlib", "xlrd", "six", "termcolor", "scikit-learn",
                      "scikit-image", "imageio", "mtcnn", "pycryptodome", "mysql-connector-python", "pymysql"],
    packages=find_packages(),
    extras_require={"gpu": ["tensorflow-gpu==1.12.0"], "cpu": ["tensorflow"]},
    scripts=["bin/make_config.sh", "bin/make_keys.sh"],
    zip_safe=False
)
