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

"""Loads libvsag_c through cffi (ABI mode) using the public C header."""

import os
import re
import sys

import cffi

_PKG_DIR = os.path.dirname(os.path.realpath(__file__))
_HEADER_NAME = "vsag_c_api.h"
EXPECTED_C_API_VERSION = 1


def _find_header():
    candidates = [
        os.path.join(_PKG_DIR, "_" + _HEADER_NAME),
        os.path.join(_PKG_DIR, "..", "..", "include", "vsag", _HEADER_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise ImportError("pyvsag cannot find the C API header " + _HEADER_NAME)


def _find_library():
    override = os.environ.get("PYVSAG_LIBRARY")
    if override:
        return override
    suffix = "dylib" if sys.platform == "darwin" else "so"
    path = os.path.join(_PKG_DIR, "libvsag_c." + suffix)
    if os.path.exists(path):
        return path
    raise ImportError(
        "pyvsag cannot find libvsag_c; set PYVSAG_LIBRARY to its path when running from a "
        "source tree"
    )


def _cdef_from_header(path):
    with open(path, encoding="utf-8") as header:
        text = header.read()
    # The C++ guards around extern "C" are not valid cdef input.
    text = re.sub(r"#ifdef __cplusplus.*?#endif", "", text, flags=re.DOTALL)
    # cffi understands integer #define constants but not other preprocessor lines.
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith("#") or re.match(r"#define\s+\w+\s+-?\d+\s*$", line)
    ]
    return "\n".join(lines)


ffi = cffi.FFI()
ffi.cdef(_cdef_from_header(_find_header()))
lib = ffi.dlopen(_find_library())

if lib.vsag_c_api_version() != EXPECTED_C_API_VERSION:
    raise ImportError(
        "pyvsag is incompatible with the loaded libvsag_c (C API version %d, expected %d)"
        % (lib.vsag_c_api_version(), EXPECTED_C_API_VERSION)
    )
