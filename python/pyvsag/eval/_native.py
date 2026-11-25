
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

"""Binds the evaluation extension of the VSAG C ABI (``vsag_eval_c_api.h``)."""

import json
import os

from .._index import _take_error
from .._lib import _PKG_DIR, _cdef_from_header, ffi, lib

_HEADER_NAME = "vsag_eval_c_api.h"


def _find_header():
    candidates = [
        os.path.join(_PKG_DIR, "_" + _HEADER_NAME),
        os.path.join(_PKG_DIR, "..", "..", "include", "vsag", _HEADER_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise ImportError("pyvsag.eval cannot find the C API header " + _HEADER_NAME)


ffi.cdef(_cdef_from_header(_find_header()))


def _check_available():
    try:
        lib.vsag_eval_build
    except AttributeError:
        raise ImportError(
            "the loaded libvsag_c does not provide the evaluation API (vsag_eval_*); "
            "rebuild pyvsag from a source tree that contains tools/eval"
        ) from None


_check_available()


def _check(err, prefix):
    code, message = _take_error(err)
    if code != lib.VSAG_SUCCESS:
        error_type = ValueError if code == lib.VSAG_INVALID_ARGUMENT else RuntimeError
        raise error_type("%s: %s" % (prefix, message))


def _config_json(config):
    if not config:
        return ffi.NULL
    try:
        return json.dumps(config).encode("utf-8")
    except TypeError as error:
        raise ValueError("evaluation options must be JSON serializable: %s" % error) from None


def _take_result(out):
    text = out[0]
    if text == ffi.NULL:
        return {}
    try:
        return json.loads(ffi.string(text).decode("utf-8"))
    finally:
        lib.vsag_string_destroy(text)


def set_num_threads_building(num_threads):
    """Set the process-wide build thread count and return the previous value."""
    previous = ffi.new("int32_t*")
    _check(lib.vsag_eval_set_num_threads_building(int(num_threads), previous), "eval failed")
    return int(previous[0])


def build(index_handle, base, metric, config):
    out = ffi.new("char**")
    err = lib.vsag_eval_build(index_handle, base.handle, metric.encode(), _config_json(config), out)
    _check(err, "eval build failed")
    return _take_result(out)


def search(index_handle, base, queries, gt_ids, gt_k, metric, config):
    out = ffi.new("char**")
    err = lib.vsag_eval_search(
        index_handle,
        ffi.NULL if base is None else base.handle,
        queries.handle,
        ffi.NULL if gt_ids is None else ffi.cast("int64_t*", gt_ids.ctypes.data),
        gt_k,
        metric.encode(),
        _config_json(config),
        out,
    )
    _check(err, "eval search failed")
    return _take_result(out)
