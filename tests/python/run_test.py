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

"""Test runner - now uses pytest under the hood"""

import os
import pytest
import sys


def run():
    """Run all tests using pytest"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = [
        "-v",
        os.path.join(script_dir, "test_ivf.py"),
        os.path.join(script_dir, "test_hnsw.py"),
        os.path.join(script_dir, "test_hgraph.py"),
        os.path.join(script_dir, "test_bruteforce.py"),
    ]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(run())
