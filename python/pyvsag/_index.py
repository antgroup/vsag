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

import json
import os

import numpy as np

from ._lib import ffi, lib

_DENSE_DTYPES = {
    # index dtype -> (numpy dtype name, description)
    "float32": ("float32", "numpy.float32"),
    "float16": ("float16", "numpy.float16"),
    "bfloat16": ("uint16", "numpy.uint16 (raw bfloat16 bits)"),
}

_CREATE_ERRORS = {
    lib.VSAG_UNSUPPORTED_INDEX: "error type: UNSUPPORTED_INDEX",
    lib.VSAG_INVALID_ARGUMENT: "error type: invalid_parameter",
}


def _take_error(err):
    """Return (code, message) of a vsag_error_t* and release it; (0, "") for NULL."""
    if err == ffi.NULL:
        return lib.VSAG_SUCCESS, ""
    code = lib.vsag_error_code(err)
    message = ffi.string(lib.vsag_error_message(err)).decode("utf-8", "replace")
    lib.vsag_error_destroy(err)
    return code, message


def _check(err, prefix):
    code, message = _take_error(err)
    if code != lib.VSAG_SUCCESS:
        raise RuntimeError("%s: %s" % (prefix, message))


def _ok(err):
    return _take_error(err)[0] == lib.VSAG_SUCCESS


def _int_array(values, dtype, name):
    """Convert to a contiguous integer array, rejecting input that would be altered."""
    array = np.asarray(values)
    if array.size == 0:
        return np.empty(array.shape, dtype=dtype)
    if array.dtype.kind not in "iu":
        raise ValueError("%s must be an integer array" % name)
    info = np.iinfo(dtype)
    if int(array.min()) < info.min or int(array.max()) > info.max:
        raise ValueError("%s has values outside the %s range" % (name, np.dtype(dtype).name))
    return np.ascontiguousarray(array, dtype=dtype)


def _parse_dtype(parameters):
    try:
        dtype = json.loads(parameters).get("dtype", "float32")
    except (ValueError, AttributeError):
        return "float32"
    return dtype if isinstance(dtype, str) else "float32"


def _ptr(array, ctype="void*"):
    return ffi.cast(ctype, array.ctypes.data)


def _is_contiguous_1d(array):
    return array.strides[0] == array.itemsize


def _int64_ids(ids, expected_size=None):
    array = _int_array(ids, np.int64, "ids")
    if array.ndim != 1:
        raise ValueError("ids must be 1-dimensional")
    if expected_size is not None and array.shape[0] != expected_size:
        raise ValueError(
            "ids length (%d) must equal num_elements (%d)" % (array.shape[0], expected_size)
        )
    return array


def _csr(index_pointers, indices, values):
    ptrs = _int_array(index_pointers, np.uint32, "index_pointers")
    idx = _int_array(indices, np.uint32, "indices")
    vals = np.ascontiguousarray(values, dtype=np.float32)
    if ptrs.ndim != 1 or idx.ndim != 1 or vals.ndim != 1:
        raise ValueError("all inputs must be 1-dimensional")
    if ptrs.shape[0] < 2:
        raise ValueError("index_pointers length must be at least 2")
    nnz = int(ptrs[-1])
    if idx.shape[0] != nnz:
        raise ValueError(
            "Size of 'indices'(%d) must equal index_pointers[last](%d)" % (idx.shape[0], nnz)
        )
    if vals.shape[0] != nnz:
        raise ValueError(
            "Size of 'values'(%d) must equal index_pointers[last](%d)" % (vals.shape[0], nnz)
        )
    if ptrs[0] != 0:
        raise ValueError("index_pointers[0] must be 0")
    bad = np.nonzero(ptrs[1:] < ptrs[:-1])[0]
    if bad.size:
        i = int(bad[0]) + 1
        raise ValueError(
            "index_pointers[%d](%d) > index_pointers[%d](%d)" % (i - 1, ptrs[i - 1], i, ptrs[i])
        )
    return ptrs, idx, vals


class _Dataset:
    """A vsag_dataset_t that borrows numpy buffers; it keeps them alive with the handle."""

    def __init__(self, num_elements, dim=None, ids=None):
        out = ffi.new("vsag_dataset_t**")
        _check(lib.vsag_dataset_create(out), "dataset failed")
        self.handle = ffi.gc(out[0], lib.vsag_dataset_destroy)
        self._keep = []
        _check(lib.vsag_dataset_set_num_elements(self.handle, num_elements), "dataset failed")
        if dim is not None:
            _check(lib.vsag_dataset_set_dim(self.handle, dim), "dataset failed")
        if ids is not None:
            self._keep.append(ids)
            _check(lib.vsag_dataset_set_ids(self.handle, _ptr(ids, "int64_t*")), "dataset failed")

    def dense(self, array):
        self._keep.append(array)
        if array.dtype == np.float32:
            err = lib.vsag_dataset_set_float32_vectors(self.handle, _ptr(array, "float*"))
        else:
            err = lib.vsag_dataset_set_float16_vectors(self.handle, _ptr(array, "uint16_t*"))
        _check(err, "dataset failed")
        return self

    def sparse(self, ptrs, idx, vals):
        self._keep.extend((ptrs, idx, vals))
        err = lib.vsag_dataset_set_sparse_vectors(
            self.handle, _ptr(ptrs, "uint32_t*"), _ptr(idx, "uint32_t*"), _ptr(vals, "float*")
        )
        _check(err, "dataset failed")
        return self


def _take_result(out):
    return ffi.gc(out[0], lib.vsag_search_result_destroy)


def _result_arrays(result):
    count = int(lib.vsag_search_result_count(result))
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    ids = np.frombuffer(
        ffi.buffer(lib.vsag_search_result_ids(result), count * 8), dtype=np.int64
    ).copy()
    dists = np.frombuffer(
        ffi.buffer(lib.vsag_search_result_distances(result), count * 4), dtype=np.float32
    ).copy()
    return ids, dists


def _statistics(result):
    return ffi.string(lib.vsag_search_result_statistics(result)).decode("utf-8")


class Index:
    """Vector search index for efficient similarity search.

    Supported dense input types:
      - float32 indexes accept numpy.float32 vectors
      - float16 indexes accept numpy.float16 vectors
      - bfloat16 indexes accept numpy.uint16 arrays containing raw bfloat16 bits

    The methods are thread-safe in the same way as the underlying C++ index; the GIL is
    released while the C++ library runs.
    """

    def __init__(self, name, parameters):
        out = ffi.new("vsag_index_t**")
        code, _ = _take_error(lib.vsag_index_create(name.encode(), parameters.encode(), out))
        if code != lib.VSAG_SUCCESS:
            raise RuntimeError(_CREATE_ERRORS.get(code, "error type: unexpectedError"))
        self._handle = ffi.gc(out[0], lib.vsag_index_destroy)
        self._dtype = _parse_dtype(parameters)

    def _dense_kind(self, operation):
        kind = _DENSE_DTYPES.get(self._dtype)
        if kind is None:
            raise ValueError(
                "%s only supports indexes with dtype in [float32, float16, bfloat16]" % operation
            )
        return kind

    def _dense_array(self, array, operation, arg_name, expected_size=0, expected_dim=0):
        np_name, description = self._dense_kind(operation)
        array = np.asarray(array)
        # Comparing dtype objects also rejects non-native byte order, which dtype.name hides.
        if array.dtype != np.dtype(np_name):
            raise ValueError("%s must be %s array" % (arg_name, description))
        if expected_size == 0:
            if array.ndim != 1:
                raise ValueError("%s must be 1-dimensional" % arg_name)
            if not _is_contiguous_1d(array):
                raise ValueError("%s must be contiguous" % arg_name)
            return array
        if array.ndim == 1:
            if not _is_contiguous_1d(array):
                raise ValueError("%s must be contiguous" % arg_name)
            size = array.shape[0]
        elif array.ndim == 2:
            if array.strides[1] != array.itemsize or (
                array.strides[0] != array.itemsize * array.shape[1]
            ):
                raise ValueError("%s must be contiguous in row-major order" % arg_name)
            if expected_dim != 0 and array.shape[1] != expected_dim:
                raise ValueError(
                    "%s second dimension (%d) must equal dim (%d)"
                    % (arg_name, array.shape[1], expected_dim)
                )
            size = array.shape[0] * array.shape[1]
        else:
            raise ValueError(
                "%s must be a 1-dimensional array or a 2-dimensional contiguous matrix"
                % arg_name
            )
        if size != expected_size:
            raise ValueError("%s size (%d) must equal %d" % (arg_name, size, expected_size))
        return array

    def _dense_dataset(self, operation, vectors, ids, num_elements, dim):
        num_elements, dim = int(num_elements), int(dim)
        array = self._dense_array(vectors, operation, "vectors", num_elements * dim, dim)
        ids = _int64_ids(ids, num_elements)
        return _Dataset(num_elements, dim, ids).dense(array)

    @staticmethod
    def _sparse_dataset(index_pointers, indices, values, ids):
        ptrs, idx, vals = _csr(index_pointers, indices, values)
        ids_arr = _int_array(ids, np.int64, "ids")
        if ids_arr.ndim != 1:
            raise ValueError("all inputs must be 1-dimensional")
        count = ptrs.shape[0] - 1
        if ids_arr.shape[0] != count:
            raise ValueError(
                "Length of 'ids'(%d) must match number of vectors(%d)" % (ids_arr.shape[0], count)
            )
        return _Dataset(count, None, ids_arr).sparse(ptrs, idx, vals)

    @staticmethod
    def _bind(args, kwargs, dense, sparse):
        """Pick the dense or sparse overload from the argument names."""
        if "index_pointers" in kwargs:
            use_sparse = True
        elif len(dense) != len(sparse):
            use_sparse = len(args) + len(kwargs) == len(sparse)
        else:
            use_sparse = len(args) >= 3 and not isinstance(args[2], (int, np.integer))
            use_sparse = use_sparse or ("values" in kwargs)
        names = sparse if use_sparse else dense
        if len(args) > len(names):
            raise TypeError("too many positional arguments")
        bound = dict(zip(names, args))
        bound.update(kwargs)
        missing = [n for n in names if n not in bound]
        extra = [n for n in bound if n not in names]
        if missing or extra:
            raise TypeError("invalid arguments (missing %s, unexpected %s)" % (missing, extra))
        return names is sparse, [bound[n] for n in names]

    _DENSE_WRITE = ("vectors", "ids", "num_elements", "dim")
    _SPARSE_WRITE = ("index_pointers", "indices", "values", "ids")

    def build(self, *args, **kwargs):
        """Build the index from dense vectors or from sparse vectors in CSR format.

        Dense:  build(vectors, ids, num_elements, dim)
        Sparse: build(index_pointers, indices, values, ids)
        """
        is_sparse, values = self._bind(args, kwargs, self._DENSE_WRITE, self._SPARSE_WRITE)
        if is_sparse:
            dataset = self._sparse_dataset(*values)
        else:
            dataset = self._dense_dataset("build", *values)
        _check(lib.vsag_index_build(self._handle, dataset.handle), "build failed")

    def add(self, *args, **kwargs):
        """Add dense vectors or sparse vectors in CSR format; same arguments as build."""
        is_sparse, values = self._bind(args, kwargs, self._DENSE_WRITE, self._SPARSE_WRITE)
        if is_sparse:
            dataset = self._sparse_dataset(*values)
        else:
            dataset = self._dense_dataset("add", *values)
        _check(lib.vsag_index_add(self._handle, dataset.handle, ffi.NULL), "add failed")

    def remove(self, ids):
        """Remove ids from the index and return the number of removed vectors."""
        ids = _int_array(ids, np.int64, "ids")
        if ids.ndim != 1:
            raise ValueError("ids must be 1-dimensional")
        removed = ffi.new("uint64_t*")
        err = lib.vsag_index_remove(self._handle, _ptr(ids, "int64_t*"), ids.shape[0], removed)
        return int(removed[0]) if _ok(err) else 0

    _DENSE_KNN = ("vector", "k", "parameters")
    _SPARSE_KNN = ("index_pointers", "indices", "values", "k", "parameters")

    def _dense_knn(self, vector, k, parameters, operation):
        """Search one vector; return (ids, dists, result, error) with -1 filled on failure."""
        vector = self._dense_array(vector, operation, "vector")
        k = int(k)
        dataset = _Dataset(1, vector.shape[0]).dense(vector)
        out = ffi.new("vsag_search_result_t**")
        err = lib.vsag_index_knn_search(
            self._handle, dataset.handle, k, parameters.encode(), ffi.NULL, out
        )
        ids = np.full(k, -1, dtype=np.int64)
        dists = np.full(k, -1.0, dtype=np.float32)
        if out[0] == ffi.NULL:
            return ids, dists, None, err
        result = _take_result(out)
        found_ids, found_dists = _result_arrays(result)
        count = min(k, found_ids.shape[0])
        ids[:count] = found_ids[:count]
        dists[:count] = found_dists[:count]
        return ids, dists, result, err

    def _flat_knn(self, vector, k, parameters):
        """Single float32 query through the allocation-free C entry point."""
        k = int(k)
        ids = np.full(k, -1, dtype=np.int64)
        dists = np.full(k, -1.0, dtype=np.float32)
        err = lib.vsag_index_knn_search_float32(
            self._handle,
            _ptr(vector, "float*"),
            1,
            vector.shape[0],
            k,
            parameters.encode(),
            ffi.NULL,
            _ptr(ids, "int64_t*"),
            _ptr(dists, "float*"),
            ffi.NULL,
        )
        return ids, dists, err

    def knn_search(self, *args, **kwargs):
        """k-nearest-neighbor search.

        Dense:  knn_search(vector, k, parameters) -> (ids, distances), each of shape (k,)
        Sparse: knn_search(index_pointers, indices, values, k, parameters)
        """
        is_sparse, values = self._bind(args, kwargs, self._DENSE_KNN, self._SPARSE_KNN)
        if is_sparse:
            return self._sparse_knn(*values, statistics=False)
        vector, k, parameters = values
        if self._dtype == "float32" and int(k) > 0:
            array = self._dense_array(vector, "knn_search", "vector")
            ids, dists, err = self._flat_knn(array, k, parameters)
            _take_error(err)
            return ids, dists
        ids, dists, _, err = self._dense_knn(vector, k, parameters, "knn_search")
        _take_error(err)
        return ids, dists

    def knn_search_with_statistics(self, *args, **kwargs):
        """Same as knn_search, but also returns the statistics JSON of the search."""
        is_sparse, values = self._bind(args, kwargs, self._DENSE_KNN, self._SPARSE_KNN)
        if is_sparse:
            return self._sparse_knn(*values, statistics=True)
        ids, dists, result, err = self._dense_knn(*values, "knn_search_with_statistics")
        _check(err, "knn search failed")
        return ids, dists, _statistics(result)

    def _sparse_knn(self, index_pointers, indices, values, k, parameters, statistics):
        ptrs, idx, vals = _csr(index_pointers, indices, values)
        k = int(k)
        count = ptrs.shape[0] - 1
        ids = np.full((count, k), -1, dtype=np.int64)
        dists = np.full((count, k), -1.0, dtype=np.float32)
        stats = []
        params = parameters.encode()
        for i in range(count):
            start, end = int(ptrs[i]), int(ptrs[i + 1])
            row_ptrs = np.array([0, end - start], dtype=np.uint32)
            dataset = _Dataset(1).sparse(row_ptrs, idx[start:end], vals[start:end])
            out = ffi.new("vsag_search_result_t**")
            err = lib.vsag_index_knn_search(self._handle, dataset.handle, k, params, ffi.NULL, out)
            if out[0] != ffi.NULL:
                result = _take_result(out)
                found_ids, found_dists = _result_arrays(result)
                n = min(k, found_ids.shape[0])
                ids[i, :n] = found_ids[:n]
                dists[i, :n] = found_dists[:n]
                if statistics:
                    stats.append(_statistics(result))
            if statistics:
                _check(err, "sparse knn search failed")
            else:
                _take_error(err)
        if statistics:
            return ids, dists, stats
        return ids, dists

    def _range(self, point, threshold, parameters, operation):
        point = self._dense_array(point, operation, "point")
        dataset = _Dataset(1, point.shape[0]).dense(point)
        out = ffi.new("vsag_search_result_t**")
        err = lib.vsag_index_range_search(
            self._handle, dataset.handle, float(threshold), -1, parameters.encode(), ffi.NULL, out
        )
        return (_take_result(out) if out[0] != ffi.NULL else None), err

    def range_search(self, point, threshold, parameters):
        """Return (ids, distances) of all vectors within the distance threshold."""
        result, err = self._range(point, threshold, parameters, "range_search")
        if not _ok(err) or result is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return _result_arrays(result)

    def range_search_with_statistics(self, point, threshold, parameters):
        """Same as range_search, but also returns the statistics JSON of the search."""
        result, err = self._range(point, threshold, parameters, "range_search_with_statistics")
        _check(err, "range search failed")
        ids, dists = _result_arrays(result)
        return ids, dists, _statistics(result)

    def calc_distances_by_id(self, query, ids):
        """Return the distances between the query and the vectors with the given ids."""
        query = np.ascontiguousarray(query, dtype=np.float32)
        ids = _int_array(ids, np.int64, "ids")
        if query.ndim != 1 or ids.ndim != 1:
            raise ValueError("query and ids must be 1-dimensional")
        if query.shape[0] == 0:
            raise RuntimeError("calc_distances_by_id failed: query must not be empty")
        dists = np.empty(ids.shape[0], dtype=np.float32)
        dataset = _Dataset(1, query.shape[0]).dense(query)
        err = lib.vsag_index_calc_distances_by_ids(
            self._handle, dataset.handle, _ptr(ids, "int64_t*"), ids.shape[0], _ptr(dists, "float*")
        )
        _check(err, "calc_distances_by_id failed")
        return dists

    cal_distance_by_id = calc_distances_by_id

    def get_num_elements(self):
        value = ffi.new("int64_t*")
        _check(lib.vsag_index_get_num_elements(self._handle, value), "get_num_elements failed")
        return int(value[0])

    def get_memory_usage(self):
        value = ffi.new("int64_t*")
        _check(lib.vsag_index_get_memory_usage(self._handle, value), "get_memory_usage failed")
        return int(value[0])

    def check_id_exist(self, id):
        exist = ffi.new("int32_t*")
        _check(lib.vsag_index_check_id_exist(self._handle, int(id), exist), "check_id_exist failed")
        return bool(exist[0])

    def get_min_max_id(self):
        """Return (min_id, max_id), or (-1, -1) if the index has no ids."""
        min_id, max_id = ffi.new("int64_t*"), ffi.new("int64_t*")
        if not _ok(lib.vsag_index_get_min_max_id(self._handle, min_id, max_id)):
            return -1, -1
        return int(min_id[0]), int(max_id[0])

    def save(self, filename):
        """Save the index to a file."""
        _check(lib.vsag_index_serialize_file(self._handle, os.fsencode(filename)), "save failed")

    def load(self, filename):
        """Load the index from a file written by save."""
        _check(lib.vsag_index_deserialize_file(self._handle, os.fsencode(filename)), "load failed")
