
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from setuptools import setup, find_packages, Extension
import os
import shutil
import platform


long_description="""VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called pyvsag. Developed by the Vector Database Team at Ant Group."""


# DON'T REMOVE: to make the wheel's name contains python version
example_module = Extension('example', sources=['example.c'])

def get_version():
    from setuptools_scm import get_version as scm_get_version
    from setuptools_scm.version import release_branch_semver, get_local_node_and_date, get_no_local_node
    from pep440 import is_canonical

    # the package with local version is not allowed to be uploaded to the
    # pypi. set build_local_version=True if you're building a local wheel.
    build_local_version = False
    local_scheme = get_no_local_node
    if build_local_version:
        local_scheme = get_local_node_and_date

    version = scm_get_version(root=f'{__file__}/../..', version_scheme=release_branch_semver, local_scheme=local_scheme)
    version_file = os.path.join(os.path.dirname(__file__), 'pyvsag', '_version.py')
    with open(version_file, 'w') as f:
        f.write(f"\n__version__ = '{version}'\n")

    # make sure the publish version is correct
    if not build_local_version and not is_canonical(version):
        print(f"!!\n\tversion {version} is incorrect, exit\n!!")
        exit(1)

    return version

setup(
    name='pyvsag',
    version=get_version(),
    description='vsag is a vector indexing library used for similarity search',
    long_description=long_description,
    url='https://github.com/antgroup/vsag',
    author='the vsag project',
    author_email='the.vsag.project@gmail.com',
    license='Apache-2.0',
    keywords='search nearest neighbors',
    install_requires=['packaging'],
    packages=find_packages(),
    package_data={
        '': ['*.so', '*.so.*'],
    },
    include_package_data=True,
    zip_safe=False,
    ext_modules=[example_module],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
