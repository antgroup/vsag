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

"""Analysis of serialized index files.

:func:`analyze_index` returns the report produced by the native ``vsag::analyze_index()``
function, which the ``pyvsag-analyze-index`` command and the C++ ``analyze_index`` tool also
call. Analysis needs only the installed wheel: no VSAG source tree, CMake or compiler.
"""

import json
import operator
import os

import _pyvsag

__all__ = ["ANALYZE_INDEX_SCHEMA_VERSION", "IndexAnalysisError", "analyze_index"]

#: Value of the ``schema_version`` member of every report.
ANALYZE_INDEX_SCHEMA_VERSION = _pyvsag.ANALYZE_INDEX_SCHEMA_VERSION


class IndexAnalysisError(RuntimeError):
    """An index file could not be analyzed.

    Attributes:
        code: Machine-readable cause, usually one of ``invalid_argument``, ``missing_file``,
            ``read_error``, ``invalid_binary``, ``unsupported_index``,
            ``unsupported_index_operation``, ``index_empty``, ``no_enough_memory``,
            ``internal_error`` or ``unknown_error``. An error raised while an index loads
            keeps its own lower-case name, such as ``dimension_not_equal``.
        message: Human-readable explanation that says how to fix the problem when possible.
    """

    def __init__(self, code, message):
        super(IndexAnalysisError, self).__init__("[{}] {}".format(code, message))
        self.code = code
        self.message = message


def _path(name, value, required=False):
    if value is None:
        if required:
            raise TypeError("{} is required".format(name))
        return ""
    try:
        path = os.fspath(value)
    except TypeError:
        raise TypeError(
            "{} must be a str or os.PathLike, not {}".format(name, type(value).__name__)
        ) from None
    return os.fsdecode(path)


def _json_object(name, value):
    if value is None:
        return "default"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    raise TypeError(
        "{} must be a JSON string, a dict or None, not {}".format(name, type(value).__name__)
    )


def analyze_index(
    index_path,
    *,
    build_parameter="default",
    query_path=None,
    query_data_type="auto",
    base_path=None,
    groundtruth_path=None,
    save_groundtruth_path=None,
    search_parameter="default",
    topk=100
):
    """Analyze a serialized index file without building or modifying it.

    The index type and parameters are read from the file. Static analysis (index metadata and
    the ``GetStats()`` statistics) always runs; dynamic analysis (``AnalyzeIndexBySearch()``)
    runs when ``query_path`` is given. The whole index is loaded into memory.

    Args:
        index_path: Serialized index file (str or os.PathLike).
        build_parameter: Build parameters (JSON string or dict) used to reload the index.
            ``"default"`` or None uses the parameters recorded in the file. Files that do not
            record them (legacy files, lazy_hgraph files and pyramid footer files) need them.
        query_path: Query dataset for dynamic analysis, or None for static analysis only.
            Dense files hold ``uint32 rows, uint32 dim, float32 data[rows * dim]``; sparse
            files hold a CSR matrix ``int64 rows, int64 cols, int64 nnz,
            int64 indptr[rows + 1], int32 indices[nnz], float32 values[nnz]``.
        query_data_type: ``"auto"`` (sparse for sparse indexes, dense otherwise),
            ``"dense"`` or ``"sparse"``.
        base_path: sindi only, sparse CSR base dataset used for exact ground truth and the
            metrics that need original vectors.
        groundtruth_path: sindi only, existing ground truth (``uint32 rows, uint32 topk,
            int32 ids[], float32 distances[]``) for dynamic analysis.
        save_groundtruth_path: sindi only, where to save the ground truth generated from
            ``base_path``.
        search_parameter: Search parameters (JSON string or dict) for dynamic analysis.
            ``"default"`` or None uses ``{"<index type>": {"ef_search": max(100, topk)}}`` for
            hgraph, lazy_hgraph and pyramid, and ``{}`` for the other index types.
        topk: Number of neighbors evaluated for each query.

    Returns:
        A dict that ``json.dumps()`` can serialize, with the members ``schema_version``
        (:data:`ANALYZE_INDEX_SCHEMA_VERSION`), ``metadata``, ``static_analysis``,
        ``dynamic_analysis`` (None without ``query_path``), ``warnings`` and
        ``compatibility``.

    Raises:
        TypeError: An argument has the wrong type.
        IndexAnalysisError: The arguments are invalid or inconsistent, an input file is
            missing, damaged or in an unsupported format, or the analysis failed. ``code``
            tells these cases apart.

    Example:
        >>> report = pyvsag.analyze_index("index.vsag", query_path="queries.bin", topk=10)
        >>> report["metadata"]["index_type"], report["dynamic_analysis"]["result"]["recall_query"]
    """
    if isinstance(topk, bool):
        raise TypeError("topk must be an int, not bool")
    try:
        topk = operator.index(topk)
    except TypeError:
        raise TypeError("topk must be an int, not {}".format(type(topk).__name__)) from None
    if not isinstance(query_data_type, str):
        raise TypeError(
            "query_data_type must be a str, not {}".format(type(query_data_type).__name__)
        )
    if not -(2 ** 63) <= topk < 2 ** 63:
        raise IndexAnalysisError("invalid_argument", "topk is out of range: {}".format(topk))

    code, payload = _pyvsag._analyze_index(
        index_path=_path("index_path", index_path, required=True),
        build_parameter=_json_object("build_parameter", build_parameter),
        query_path=_path("query_path", query_path),
        query_data_type=query_data_type,
        base_path=_path("base_path", base_path),
        groundtruth_path=_path("groundtruth_path", groundtruth_path),
        save_groundtruth_path=_path("save_groundtruth_path", save_groundtruth_path),
        search_parameter=_json_object("search_parameter", search_parameter),
        topk=topk,
    )
    if code is not None:
        raise IndexAnalysisError(code, payload.decode("utf-8", "replace"))
    return json.loads(payload.decode("utf-8"))
