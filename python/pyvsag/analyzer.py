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

"""Public Python API for inspecting serialized VSAG index files.

The implementation is shared with the C++ ``analyze_index`` executable: both call the same
native analyzer component, so loading and analysis behavior cannot drift between interfaces.
"""

import json

from ._index import _take_error
from ._lib import ffi, lib

__all__ = ["ANALYSIS_SCHEMA_VERSION", "analyze_index"]

#: Schema version of the dictionary returned by :func:`analyze_index`.
ANALYSIS_SCHEMA_VERSION = "1.0"


def _optional_path(value, name):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("{} must be a string or None".format(name))
    return value


def _analyze_index_json(
    index_path,
    build_parameter,
    query_path,
    query_data_type,
    base_path,
    groundtruth_path,
    save_groundtruth_path,
    search_parameter,
    topk,
):
    out_json = ffi.new("char**")
    error = lib.vsag_analyze_index(
        index_path.encode("utf-8"),
        build_parameter.encode("utf-8"),
        query_path.encode("utf-8"),
        query_data_type.encode("utf-8"),
        base_path.encode("utf-8"),
        groundtruth_path.encode("utf-8"),
        save_groundtruth_path.encode("utf-8"),
        search_parameter.encode("utf-8"),
        topk,
        out_json,
    )
    code, message = _take_error(error)
    if code == lib.VSAG_INVALID_ARGUMENT:
        raise ValueError(message)
    if code != lib.VSAG_SUCCESS:
        raise RuntimeError(message or "index analysis failed")
    if out_json[0] == ffi.NULL:
        raise RuntimeError("index analysis returned no JSON result")

    try:
        return ffi.string(out_json[0]).decode("utf-8")
    finally:
        lib.vsag_string_destroy(out_json[0])


def analyze_index(
    index_path,
    build_parameter="default",
    query_path=None,
    query_data_type="auto",
    base_path=None,
    groundtruth_path=None,
    save_groundtruth_path=None,
    search_parameter="default",
    topk=100,
):
    """Analyze a serialized VSAG index.

    Without ``query_path`` this performs static analysis only and returns index metadata plus the
    ``GetStats()`` statistics. Supplying ``query_path`` additionally runs
    ``AnalyzeIndexBySearch`` and returns the dynamic metrics.

    Args:
        index_path (str): Path to a serialized VSAG index file.
        build_parameter (str): Build parameter JSON used to reload the index. Use ``"default"``
            (the default) to reuse the parameters embedded in the index file.
        query_path (str): Binary query dataset path. ``None`` or an empty string disables
            dynamic analysis.
        query_data_type (str): One of ``"auto"``, ``"dense"`` or ``"sparse"``. ``"auto"`` selects
            sparse loading for SINDI indexes and dense loading otherwise.
        base_path (str): Optional sparse CSR base dataset used by SINDI analysis and
            ground-truth generation.
        groundtruth_path (str): Optional existing ground-truth file reused by dynamic analysis.
        save_groundtruth_path (str): Optional output path for generated ground truth.
        search_parameter (str): Search parameter JSON used by dynamic analysis, or ``"default"``.
        topk (int): Top-K used by dynamic analysis. Must be positive.

    Returns:
        dict: A JSON-serializable report with these top-level keys:

        - ``schema_version``: result layout version.
        - ``index``: index metadata (path, type, dimension, data type, metric, ...).
        - ``static_analysis``: ``GetStats()`` statistics.
        - ``dynamic_analysis``: query-based results, or ``None`` when no query was supplied.
        - ``warnings``: non-fatal compatibility or configuration notices.
        - ``compatibility``: supported-feature and metadata-completeness information.

    Raises:
        TypeError: A ``str``/``None`` path argument has the wrong type, or ``topk`` is not an int.
        ValueError: An argument value is invalid or the requested combination is unsupported.
        RuntimeError: The index or dataset cannot be loaded, deserialized or analyzed.

    Example:
        >>> import pyvsag
        >>> report = pyvsag.analyze_index(index_path="/tmp/my_index.hgraph")
        >>> report["schema_version"]
        '1.0'
        >>> import json
        >>> json.dumps(report)  # the report is JSON-serializable
        '...'
    """
    if not isinstance(index_path, str):
        raise TypeError("index_path must be a string")
    if not index_path:
        raise ValueError("index_path must not be empty")
    if not isinstance(build_parameter, str):
        raise TypeError("build_parameter must be a JSON string")
    if not isinstance(query_data_type, str):
        raise TypeError("query_data_type must be a string")
    if not isinstance(search_parameter, str):
        raise TypeError("search_parameter must be a JSON string")
    if isinstance(topk, bool) or not isinstance(topk, int):
        raise TypeError("topk must be an int")
    if topk <= 0:
        raise ValueError("topk must be a positive integer")

    payload = _analyze_index_json(
        index_path=index_path,
        build_parameter=build_parameter,
        query_path=_optional_path(query_path, "query_path"),
        query_data_type=query_data_type,
        base_path=_optional_path(base_path, "base_path"),
        groundtruth_path=_optional_path(groundtruth_path, "groundtruth_path"),
        save_groundtruth_path=_optional_path(
            save_groundtruth_path, "save_groundtruth_path"
        ),
        search_parameter=search_parameter,
        topk=topk,
    )
    return json.loads(payload)
