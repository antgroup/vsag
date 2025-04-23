

from __future__ import print_function
from setuptools import setup, find_packages
import os
import shutil
import platform


long_description="""VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called pyvsag. Developed by the Vector Database Team at Ant Group."""

setup(
    name='pyvsag',
    version='0.0.11',
    description='vsag is a vector indexing library used for similarity search',
    long_description=long_description,
    url='https://github.com/alipay/vsag',
    author='the vsag project',
    author_email='the.vsag.project@gmail.com',
    license='Apache-2.0',
    keywords='search nearest neighbors',
    install_requires=['packaging'],
    packages=find_packages(),
    pacakge_dir={'':'.'},
    package_data={
        '': ['*.so', '*.so.*'],
    },
    include_package_data=True,
    zip_safe=False,
)
