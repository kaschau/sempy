#!/usr/bin/env python

import os
import re

from setuptools import setup

# Get version
vfile = open("./sempy/_version.py").read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)

if not os.path.exists("build"):
    os.makedirs("build")
else:
    try:
        os.unlink("build/animateAlphas")
    except FileNotFoundError:
        pass
    try:
        os.unlink("build/semForPEREGRINE")
    except FileNotFoundError:
        pass
os.symlink("../bin/animateAlphas.py", "build/animateAlphas")
os.symlink("../bin/semForPEREGRINE.py", "build/semForPEREGRINE")

# Hard dependencies
install_requires = [
    "mpi4py >= 3.0",
    "numpy >= 1.20",
    "numba >= 0.55.1",
    "scipy >= 1.5",
    "pyyaml >= 6.0",
]

long_description = """
ensemble synthetic eddy method
"""

setup(
    name="sempy",
    version=version,
    author="Kyle Schau",
    author_email="ksachau89@gmail.com",
    description="Ensemble Synthetic Eddy Method",
    package_dir={"": "./"},
    long_description=long_description,
    install_requires=install_requires,
    scripts=["build/animateAlphas", "build/semForPEREGRINE"],
    python_requires=">=3.8",
    test_suite="tests",
    zip_safe=False,
)
