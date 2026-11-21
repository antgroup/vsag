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

"""In-memory datasets for :mod:`pyvsag.eval`."""

import numpy as np

from .._index import _Dataset, _int_array

METRICS = ("l2", "ip", "cosine")

# Values of the ``distance`` attribute used by the ann-benchmarks / eval_performance HDF5 files.
_HDF5_METRICS = {"euclidean": "l2", "angular": "cosine", "ip": "ip", "l2": "l2", "cosine": "cosine"}

_GROUND_TRUTH_BATCH_BYTES = 256 * 1024 * 1024


def _check_metric(metric):
    if metric not in METRICS:
        raise ValueError("metric must be one of %s, got %r" % (list(METRICS), metric))
    return metric


def _float32_matrix(array, name):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("%s must be a non-empty 2-dimensional array" % name)
    if array.dtype.kind not in "fiu":
        raise ValueError("%s must be a numeric array" % name)
    return np.ascontiguousarray(array, dtype=np.float32)


def compute_ground_truth(base, queries, k, metric="l2", ids=None):
    """Exact top-``k`` neighbours of every query by brute force with numpy.

    Returns ``(ids, distances)`` with shapes ``(num_queries, k)``. ``ids`` default to the row
    numbers of ``base``. Distances are squared L2 for ``"l2"``, ``1 - <q, b>`` for ``"ip"`` and
    cosine distance for ``"cosine"``, the same ordering VSAG uses.
    """
    _check_metric(metric)
    base = _float32_matrix(base, "base")
    queries = _float32_matrix(queries, "queries")
    if base.shape[1] != queries.shape[1]:
        raise ValueError("base and queries must have the same dim")
    k = int(k)
    if k <= 0 or k > base.shape[0]:
        raise ValueError("k must be between 1 and the number of base vectors")
    row_ids = _int_array(np.arange(base.shape[0]) if ids is None else ids, np.int64, "ids")
    if row_ids.shape != (base.shape[0],):
        raise ValueError("ids must have one entry per base vector")

    if metric == "cosine":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    base_norms = np.einsum("ij,ij->i", base, base) if metric == "l2" else None

    batch = max(1, _GROUND_TRUTH_BATCH_BYTES // (4 * base.shape[0]))
    out_ids = np.empty((queries.shape[0], k), dtype=np.int64)
    out_dists = np.empty((queries.shape[0], k), dtype=np.float32)
    for begin in range(0, queries.shape[0], batch):
        chunk = queries[begin : begin + batch]
        product = chunk @ base.T
        if metric == "l2":
            dists = np.einsum("ij,ij->i", chunk, chunk)[:, None] - 2.0 * product + base_norms
            np.maximum(dists, 0.0, out=dists)
        else:
            dists = 1.0 - product
        top = np.argpartition(dists, k - 1, axis=1)[:, :k]
        top_dists = np.take_along_axis(dists, top, axis=1)
        order = np.argsort(top_dists, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        out_ids[begin : begin + batch] = row_ids[top]
        out_dists[begin : begin + batch] = np.take_along_axis(top_dists, order, axis=1)
    return out_ids, out_dists


class EvalDataset:
    """float32 base/query vectors plus optional exact ground truth, held in memory.

    Build it with :meth:`from_arrays` or :meth:`from_hdf5`. The numpy buffers are borrowed by the
    native evaluation during a call and are never modified.
    """

    def __init__(self, base, queries, ground_truth, ids, metric):
        self.base = base
        self.queries = queries
        self.ground_truth = ground_truth
        self.ids = ids
        self.metric = metric

    @classmethod
    def from_arrays(cls, base, queries, ground_truth=None, *, metric="l2", ids=None):
        """Create a dataset from numpy arrays.

        Args:
            base: ``(N, dim)`` vectors to build the index from.
            queries: ``(Q, dim)`` query vectors.
            ground_truth: optional ``(Q, K)`` integer matrix with the ids of the exact nearest
                neighbours of every query, nearest first. Needed for recall.
            metric: ``"l2"``, ``"ip"`` or ``"cosine"``.
            ids: optional ``(N,)`` int64 ids of the base vectors (default ``0..N-1``). The ids in
                ``ground_truth`` refer to them.
        """
        _check_metric(metric)
        base = _float32_matrix(base, "base")
        queries = _float32_matrix(queries, "queries")
        if base.shape[1] != queries.shape[1]:
            raise ValueError(
                "base dim (%d) and queries dim (%d) differ" % (base.shape[1], queries.shape[1])
            )
        ids = _int_array(np.arange(base.shape[0]) if ids is None else ids, np.int64, "ids")
        if ids.shape != (base.shape[0],):
            raise ValueError("ids must be 1-dimensional with one entry per base vector")
        if ground_truth is not None:
            ground_truth = _int_array(ground_truth, np.int64, "ground_truth")
            if ground_truth.ndim != 2 or ground_truth.shape[0] != queries.shape[0]:
                raise ValueError("ground_truth must have shape (num_queries, k)")
            if ground_truth.shape[1] == 0:
                raise ValueError("ground_truth must contain at least one neighbour per query")
        return cls(base, queries, ground_truth, ids, metric)

    @classmethod
    def from_hdf5(cls, path, *, metric=None):
        """Load an ann-benchmarks style HDF5 file (also the ones used by ``eval_performance``).

        Reads the ``train``, ``test`` and ``neighbors`` datasets and the ``distance`` attribute.
        Requires the optional ``h5py`` package (``pip install pyvsag[eval]``). Only dense
        float vectors are supported; sparse and multi-vector files are not.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "reading HDF5 datasets requires h5py; install it with 'pip install pyvsag[eval]'"
            ) from None

        with h5py.File(path, "r") as file:
            kind = file.attrs.get("type", "dense")
            kind = kind.decode("utf-8") if isinstance(kind, bytes) else str(kind)
            if kind != "dense":
                raise ValueError("unsupported HDF5 dataset type %r: only dense is supported" % kind)
            for name in ("train", "test"):
                if name not in file:
                    raise ValueError("missing required HDF5 dataset %r" % name)
            if metric is None:
                attribute = file.attrs.get("distance", "euclidean")
                attribute = (
                    attribute.decode("utf-8") if isinstance(attribute, bytes) else str(attribute)
                )
                if attribute not in _HDF5_METRICS:
                    raise ValueError("unsupported HDF5 distance %r" % attribute)
                metric = _HDF5_METRICS[attribute]
            base = file["train"][...]
            queries = file["test"][...]
            neighbors = file["neighbors"][...] if "neighbors" in file else None
        return cls.from_arrays(base, queries, neighbors, metric=metric)

    @property
    def num_base(self):
        return self.base.shape[0]

    @property
    def num_queries(self):
        return self.queries.shape[0]

    @property
    def dim(self):
        return self.base.shape[1]

    def with_ground_truth(self, k):
        """Return a copy whose ground truth is computed exactly by brute force."""
        ids, _ = compute_ground_truth(self.base, self.queries, k, self.metric, self.ids)
        return EvalDataset(self.base, self.queries, ids, self.ids, self.metric)

    def _base_dataset(self):
        return _Dataset(self.num_base, self.dim, self.ids).dense(self.base)

    def _queries_dataset(self):
        return _Dataset(self.num_queries, self.dim).dense(self.queries)

    def __repr__(self):
        ground_truth = "None" if self.ground_truth is None else str(self.ground_truth.shape)
        return "EvalDataset(base=%s, queries=%s, ground_truth=%s, metric=%r)" % (
            self.base.shape,
            self.queries.shape,
            ground_truth,
            self.metric,
        )
