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

"""Deterministic dataset profiling.

The profiler makes a single streaming pass over the selected split for
full-dataset statistics and a bounded random sample for distribution,
pairwise-distance, anisotropy, and optional clustering statistics.  No
all-pairs ``O(N^2)`` matrix is ever allocated for the complete dataset: pairwise
work is capped by ``pairwise_sample_size``.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hdf5 import DEFAULT_CHUNK_BYTES, HDF5Dataset, DatasetMetadata
from .report import DEFINITIONS, build_report, metric_entry
from .sampling import SAMPLING_METHOD, make_rng, sample_indices
from .stats import (
    HISTOGRAM_BINS,
    as_finite_float,
    downsample,
    histogram,
    summarize,
    top_indices,
)

VALID_METRICS = ("euclidean", "ip", "angular")
_DEFAULT_DENSE_METRIC = "euclidean"
_DEFAULT_SPARSE_METRIC = "ip"
_MAX_ANISOTROPY_DIM = 50000
_MAX_ANISOTROPY_BYTES = 256 * 1024 * 1024


@dataclass
class ProfilerConfig:
    """Configuration for :func:`profile_dataset`."""

    seed: int = 42
    sample_size: int = 1024
    pairwise_sample_size: int = 512
    metric: Optional[str] = None
    clusters: int = 0
    near_duplicate_relative_epsilon: float = 1e-3
    max_chunk_bytes: int = DEFAULT_CHUNK_BYTES
    max_dimension_points: int = 1024

    def validate(self) -> None:
        if int(self.sample_size) <= 0:
            raise ValueError("sample_size must be positive")
        if int(self.pairwise_sample_size) <= 0:
            raise ValueError("pairwise_sample_size must be positive")
        if int(self.clusters) < 0:
            raise ValueError("clusters must be non-negative")
        if int(self.clusters) > 1000:
            raise ValueError("clusters must be at most 1000")
        if not math.isfinite(float(self.near_duplicate_relative_epsilon)):
            raise ValueError("near_duplicate_relative_epsilon must be finite")
        if float(self.near_duplicate_relative_epsilon) < 0:
            raise ValueError("near_duplicate_relative_epsilon must be non-negative")
        if int(self.max_chunk_bytes) < 16:
            raise ValueError("max_chunk_bytes is too small")
        if int(self.max_dimension_points) <= 0:
            raise ValueError("max_dimension_points must be positive")


def profile_dataset(
    path: str,
    split: str = "train",
    *,
    seed: int = 42,
    sample_size: int = 1024,
    pairwise_sample_size: int = 512,
    metric: Optional[str] = None,
    clusters: int = 0,
    near_duplicate_relative_epsilon: float = 1e-3,
    max_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_dimension_points: int = 1024,
) -> Dict[str, Any]:
    """Profile one split of a dense or sparse VSAG-compatible HDF5 file.

    Args:
        path: Path to the ``.hdf5`` file.
        split: Dataset to profile (``"train"`` by default).
        seed: Explicit seed for every sampling operation.
        sample_size: Maximum number of vectors drawn for distribution and
            dimensionality-reduction metrics.
        pairwise_sample_size: Maximum number of sampled vectors used for the
            pairwise-distance matrix.  Must not exceed the complete dataset.
        metric: ``"euclidean"``, ``"ip"``, or ``"angular"``.  When omitted the
            file's ``distance`` attribute is used, falling back to
            ``"euclidean"`` for dense and ``"ip"`` for sparse files.
        clusters: Optional k-means cluster count on the PCA-2 projection.
        near_duplicate_relative_epsilon: Relative pairwise-distance threshold
            used for the near-duplicate indicator.
        max_chunk_bytes: Streaming read chunk size.
        max_dimension_points: Maximum per-dimension points stored in reports;
            longer arrays are bucket-averaged.

    Returns:
        A JSON-serializable versioned report dictionary.

    Raises:
        ValueError: For invalid configuration or unsupported/corrupt input.
    """
    config = ProfilerConfig(
        seed=seed,
        sample_size=sample_size,
        pairwise_sample_size=pairwise_sample_size,
        metric=metric,
        clusters=clusters,
        near_duplicate_relative_epsilon=near_duplicate_relative_epsilon,
        max_chunk_bytes=max_chunk_bytes,
        max_dimension_points=max_dimension_points,
    )
    config.validate()

    with HDF5Dataset(path, split=split) as dataset:
        metadata = dataset.metadata
        warnings: List[str] = list(metadata.warnings)
        metric_name = _normalize_metric(config.metric, metadata, warnings)
        rng = make_rng(config.seed)

        if metadata.vector_type == "dense":
            metrics = _profile_dense(dataset, config, metric_name, rng, warnings)
        else:
            metrics = _profile_sparse(dataset, config, metric_name, rng, warnings)

        requested_sample = int(config.sample_size)
        requested_pairwise = int(config.pairwise_sample_size)
        effective_sample = min(requested_sample, int(metadata.count or 0))
        effective_pairwise = min(requested_pairwise, effective_sample)
        if effective_pairwise < requested_pairwise:
            warnings.append(
                "pairwise_sample_size reduced from %d to %d to stay within the "
                "available sample/dataset size"
                % (requested_pairwise, effective_pairwise)
            )

        report_config = {
            "seed": int(config.seed),
            "sample_size": requested_sample,
            "pairwise_sample_size": requested_pairwise,
            "effective_sample_size": effective_sample,
            "effective_pairwise_sample_size": effective_pairwise,
            "sampling_method": SAMPLING_METHOD,
            "pairwise_sampling_method": (
                "first effective_pairwise_sample_size rows of the seeded random "
                "sample (which is itself uniformly random)"
            ),
            "metric": metric_name,
            "clusters": int(config.clusters),
            "near_duplicate_relative_epsilon": float(
                config.near_duplicate_relative_epsilon
            ),
            "max_dimension_points": int(config.max_dimension_points),
        }
        return build_report(metadata, report_config, metrics, warnings)


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------
def _normalize_metric(
    requested: Optional[str], metadata: DatasetMetadata, warnings: List[str]
) -> str:
    aliases = {
        "l2": "euclidean",
        "euclid": "euclidean",
        "inner_product": "ip",
        "dot": "ip",
        "cosine": "angular",
        "cosine_similarity": "angular",
    }
    candidate = requested if requested is not None else metadata.distance
    if candidate is None:
        fallback = _DEFAULT_SPARSE_METRIC if metadata.vector_type == "sparse" else _DEFAULT_DENSE_METRIC
        warnings.append(
            "no distance metric was provided and the file has no 'distance' attribute; "
            "using %r" % fallback
        )
        return fallback
    normalized = aliases.get(str(candidate).strip().lower(), str(candidate).strip().lower())
    if normalized not in VALID_METRICS:
        raise ValueError(
            "unsupported distance metric %r (expected one of %s)"
            % (candidate, ", ".join(VALID_METRICS))
        )
    if metadata.vector_type == "sparse" and normalized != "ip":
        warnings.append(
            "sparse VSAG evaluation supports only the 'ip' metric; %r is computed "
            "here for profiling but is not a supported eval metric for sparse files"
            % normalized
        )
    return normalized


# ---------------------------------------------------------------------------
# Dense profiling
# ---------------------------------------------------------------------------
def _profile_dense(
    dataset: HDF5Dataset,
    config: ProfilerConfig,
    metric: str,
    rng: np.random.Generator,
    warnings: List[str],
) -> Dict[str, Any]:
    metadata = dataset.metadata
    count = int(metadata.count or 0)
    dim = int(metadata.dim or 0)
    sample_idx = sample_indices(count, min(config.sample_size, count), rng)
    pair_n = min(config.pairwise_sample_size, sample_idx.size)

    dim_count = np.zeros(dim, dtype=np.float64)
    dim_sum = np.zeros(dim, dtype=np.float64)
    dim_sumsq = np.zeros(dim, dtype=np.float64)
    dim_zero = np.zeros(dim, dtype=np.float64)
    dim_min = np.full(dim, np.inf, dtype=np.float64)
    dim_max = np.full(dim, -np.inf, dtype=np.float64)

    hashes: Counter = Counter()
    norms_parts: List[np.ndarray] = []
    total_values = 0
    zero_values = 0
    invalid_values = 0
    vectors_with_invalid = 0
    all_zero_vectors = 0

    for _, block in dataset.iter_dense_blocks(block_rows=65536):
        raw = np.asarray(block)
        if raw.dtype.kind == "f":
            finite = np.isfinite(raw)
        else:
            finite = np.ones(raw.shape, dtype=bool)
        valid = np.where(finite, raw, 0).astype(np.float64, copy=False)
        invalid_in_block = int((~finite).sum())
        invalid_values += invalid_in_block
        if invalid_in_block:
            vectors_with_invalid += int((~finite).any(axis=1).sum())
        total_values += int(raw.size)
        zero_values += int(np.count_nonzero((raw == 0) & finite))

        norms = np.sqrt(np.einsum("ij,ij->i", valid, valid))
        norms_parts.append(norms)
        all_zero_vectors += int(np.count_nonzero(norms == 0))

        dim_count += finite.sum(axis=0)
        dim_sum += valid.sum(axis=0)
        dim_sumsq += np.einsum("ij,ij->j", valid, valid)
        dim_zero += ((raw == 0) & finite).sum(axis=0)
        masked_min = np.where(finite, raw, np.inf).min(axis=0)
        masked_max = np.where(finite, raw, -np.inf).max(axis=0)
        dim_min = np.minimum(dim_min, masked_min)
        dim_max = np.maximum(dim_max, masked_max)

        for row in raw:
            contiguous = np.ascontiguousarray(row)
            hashes[_digest(contiguous.tobytes())] += 1

    norms = (
        np.concatenate(norms_parts)
        if norms_parts
        else np.empty(0, dtype=np.float64)
    )
    sample_matrix = dataset.read_dense_rows(sample_idx).astype(np.float64, copy=True)
    # Invalid (NaN/Inf) entries are excluded from numeric summaries and treated
    # as zero, matching the semantics of the norms metric.
    sample_matrix = np.where(np.isfinite(sample_matrix), sample_matrix, 0.0)
    pair_matrix = sample_matrix[:pair_n]
    pair_values = _upper_triangle(_pairwise_dense(pair_matrix, metric))

    metrics: Dict[str, Any] = {
        "invalid_values": _invalid_metric(
            invalid_values, vectors_with_invalid, total_values, count
        ),
        "norms": _norms_metric(norms),
        "outliers": _outliers_metric(norms),
        "dimensions": _dimensions_metric(
            dim_count,
            dim_sum,
            dim_sumsq,
            dim_zero,
            dim_min,
            dim_max,
            config.max_dimension_points,
            warnings,
        ),
        "sparsity": _dense_sparsity_metric(
            total_values, zero_values, all_zero_vectors, count
        ),
        "duplicates": _duplicates_metric(
            hashes, count, pair_values, config.near_duplicate_relative_epsilon
        ),
    }
    metrics.update(
        _sampled_metrics(
            pair_matrix=pair_matrix,
            sample_matrix=sample_matrix,
            pair_values=pair_values,
            metric=metric,
            config=config,
            rng=rng,
            warnings=warnings,
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# Sparse profiling
# ---------------------------------------------------------------------------
def _profile_sparse(
    dataset: HDF5Dataset,
    config: ProfilerConfig,
    metric: str,
    rng: np.random.Generator,
    warnings: List[str],
) -> Dict[str, Any]:
    metadata = dataset.metadata
    if metadata.count is None:
        counted = 0
        for _ in dataset.iter_sparse_records(max_chunk_bytes=config.max_chunk_bytes):
            counted += 1
        metadata.count = counted
        warnings.append(
            "sparse file has no '%s_offsets'; the byte stream was scanned once to "
            "count records before profiling" % metadata.split
        )
    count = int(metadata.count or 0)
    if count <= 0:
        raise ValueError("sparse dataset '/%s' is empty" % metadata.split)
    sample_idx = sample_indices(count, min(config.sample_size, count), rng)
    pair_n = min(config.pairwise_sample_size, sample_idx.size)
    sample_set = set(int(index) for index in sample_idx)

    accumulator = _DimAccumulator()
    hashes: Counter = Counter()
    norms_parts: List[np.ndarray] = []
    nnz_parts: List[np.ndarray] = []
    stored_entries = 0
    nonzero_entries = 0
    invalid_values = 0
    vectors_with_invalid = 0
    all_zero_vectors = 0
    sample_records: List[Tuple[int, np.ndarray, np.ndarray]] = []

    for index, raw_ids, raw_values in dataset.iter_sparse_records(
        max_chunk_bytes=config.max_chunk_bytes
    ):
        if raw_ids.size:
            order = np.argsort(raw_ids, kind="stable")
            ids = raw_ids[order].astype(np.uint32, copy=False)
            values = raw_values[order].astype(np.float32, copy=False)
        else:
            ids = raw_ids
            values = raw_values
        length = int(ids.size)
        stored_entries += length
        finite = np.isfinite(values)
        valid = np.where(finite, values, 0).astype(np.float64, copy=False)
        invalid_values += int((~finite).sum())
        if not bool(finite.all()):
            vectors_with_invalid += 1
        nonzero_entries += int(np.count_nonzero(valid))
        norms = float(math.sqrt(float(np.dot(valid, valid))))
        norms_parts.append(np.array([norms], dtype=np.float64))
        nnz_parts.append(np.array([length], dtype=np.int64))
        if norms == 0.0:
            all_zero_vectors += 1
        accumulator.add(ids, values, finite)
        hashes[_digest(ids.astype("<u4", copy=False).tobytes() + values.astype("<f4", copy=False).tobytes())] += 1
        if index in sample_set:
            sample_records.append((index, ids.copy(), values.copy()))

    if len(sample_records) != len(sample_set):
        raise ValueError("sparse sampling failed: not all sampled records were found")
    sample_records.sort(key=lambda item: item[0])
    pair_records = sample_records[:pair_n]
    dim = accumulator.dim
    metadata.dim = dim
    if dim == 0:
        warnings.append(
            "all sparse vectors are empty; dimensionality, density, and "
            "per-dimension statistics are unavailable"
        )

    dim_count, dim_sum, dim_sumsq, dim_zero, dim_min, dim_max = accumulator.arrays()
    norms = np.concatenate(norms_parts) if norms_parts else np.empty(0)
    nnz = np.concatenate(nnz_parts) if nnz_parts else np.empty(0, dtype=np.int64)

    pair_matrix = _pairwise_sparse(pair_records, metric)
    pair_values = _upper_triangle(pair_matrix)
    sample_matrix = _sparse_sample_to_dense(sample_records, dim)

    metrics: Dict[str, Any] = {
        "invalid_values": _invalid_metric(
            invalid_values, vectors_with_invalid, count * dim if dim else None, count
        ),
        "norms": _norms_metric(norms),
        "outliers": _outliers_metric(norms),
        "dimensions": _dimensions_metric(
            dim_count,
            dim_sum,
            dim_sumsq,
            dim_zero,
            dim_min,
            dim_max,
            config.max_dimension_points,
            warnings,
        ),
        "sparsity": _sparse_sparsity_metric(
            count,
            dim,
            stored_entries,
            nonzero_entries,
            nnz,
            all_zero_vectors,
        ),
        "duplicates": _duplicates_metric(
            hashes, count, pair_values, config.near_duplicate_relative_epsilon
        ),
    }
    metrics.update(
        _sampled_metrics(
            pair_matrix=pair_matrix,
            sample_matrix=sample_matrix,
            pair_values=pair_values,
            metric=metric,
            config=config,
            rng=rng,
            warnings=warnings,
            sparse_pairwise=True,
        )
    )
    return metrics


class _DimAccumulator:
    """Growable per-dimension accumulators for streaming sparse data."""

    def __init__(self) -> None:
        self._max_index = -1
        self.count = np.zeros(0, dtype=np.float64)
        self.sum = np.zeros(0, dtype=np.float64)
        self.sumsq = np.zeros(0, dtype=np.float64)
        self.zero = np.zeros(0, dtype=np.float64)
        self.min = np.full(0, np.inf, dtype=np.float64)
        self.max = np.full(0, -np.inf, dtype=np.float64)

    @property
    def dim(self) -> int:
        return self._max_index + 1

    def _ensure(self, size: int) -> None:
        if size <= self.count.size:
            return
        new_size = max(size, max(16, self.count.size * 2))
        pad = new_size - self.count.size
        self.count = np.concatenate([self.count, np.zeros(pad)])
        self.sum = np.concatenate([self.sum, np.zeros(pad)])
        self.sumsq = np.concatenate([self.sumsq, np.zeros(pad)])
        self.zero = np.concatenate([self.zero, np.zeros(pad)])
        self.min = np.concatenate([self.min, np.full(pad, np.inf)])
        self.max = np.concatenate([self.max, np.full(pad, -np.inf)])

    def add(self, ids: np.ndarray, values: np.ndarray, finite: np.ndarray) -> None:
        if ids.size == 0:
            return
        self._max_index = max(self._max_index, int(ids.max()))
        self._ensure(self._max_index + 1)
        if not np.any(finite):
            return
        index = ids[finite].astype(np.int64, copy=False)
        vals = values[finite].astype(np.float64, copy=False)
        np.add.at(self.count, index, 1.0)
        np.add.at(self.sum, index, vals)
        np.add.at(self.sumsq, index, vals * vals)
        zero = vals == 0
        if np.any(zero):
            np.add.at(self.zero, index[zero], 1.0)
        np.minimum.at(self.min, index, vals)
        np.maximum.at(self.max, index, vals)

    def arrays(self):
        dim = self.dim
        return (
            self.count[:dim],
            self.sum[:dim],
            self.sumsq[:dim],
            self.zero[:dim],
            self.min[:dim],
            self.max[:dim],
        )


# ---------------------------------------------------------------------------
# Shared metric builders
# ---------------------------------------------------------------------------
def _norms_metric(norms: np.ndarray) -> Dict[str, Any]:
    stats = summarize(norms)
    return metric_entry(
        DEFINITIONS["norms"],
        "full",
        None,
        count=int(norms.size),
        stats=stats,
        histogram=histogram(norms),
        non_finite_norms=int(np.size(norms) - np.count_nonzero(np.isfinite(norms))),
    )


def _outliers_metric(norms: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(norms, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return metric_entry(
            DEFINITIONS["outliers"],
            "full",
            None,
            count=0,
            outlier_count=0,
            outlier_ratio=None,
            median_norm=None,
            mad_norm=None,
            max_over_median_norm=None,
            method="n/a",
        )
    median = float(np.median(values))
    deviation = np.abs(values - median)
    mad = float(np.median(deviation))
    method = "modified z-score (median/MAD)"
    if mad > 0:
        modified_z = 0.6745 * (values - median) / mad
        mask = np.abs(modified_z) > 3.5
        threshold = 3.5 * mad / 0.6745
    else:
        q1, q3 = np.percentile(values, [25, 75], method="linear")
        iqr = float(q3 - q1)
        if iqr > 0:
            mask = (values < q1 - 3.0 * iqr) | (values > q3 + 3.0 * iqr)
            method = "IQR fallback (MAD is zero)"
            threshold = 3.0 * iqr
        else:
            mask = np.zeros(values.shape, dtype=bool)
            method = "all norms identical; no outlier can be distinguished"
            threshold = 0.0
    outlier_count = int(np.count_nonzero(mask))
    max_over_median = float(values.max() / median) if median > 0 else None
    return metric_entry(
        DEFINITIONS["outliers"],
        "full",
        None,
        count=int(values.size),
        outlier_count=outlier_count,
        outlier_ratio=as_finite_float(outlier_count / float(values.size)),
        median_norm=as_finite_float(median),
        mad_norm=as_finite_float(mad),
        threshold=as_finite_float(threshold),
        max_over_median_norm=as_finite_float(max_over_median),
        method=method,
    )


def _invalid_metric(
    invalid_values: int,
    vectors_with_invalid: int,
    total_values: Optional[int],
    count: int,
) -> Dict[str, Any]:
    total = int(total_values) if total_values is not None else None
    ratio = None
    if total:
        ratio = invalid_values / float(total)
    return metric_entry(
        DEFINITIONS["invalid_values"],
        "full",
        None,
        count=count,
        total_values=total,
        invalid_values=int(invalid_values),
        vectors_with_invalid=int(vectors_with_invalid),
        invalid_ratio=as_finite_float(ratio),
        vectors_with_invalid_ratio=as_finite_float(
            vectors_with_invalid / float(count) if count else None
        ),
    )


def _dimensions_metric(
    count: np.ndarray,
    total: np.ndarray,
    total_sq: np.ndarray,
    zero: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    max_points: int,
    warnings: List[str],
) -> Dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        variance = np.where(count > 0, total_sq / count - mean * mean, np.nan)
        std = np.sqrt(np.maximum(variance, 0))
        zero_ratio = np.where(count > 0, zero / count, np.nan)
    minimum = np.where(np.isfinite(minimum), minimum, np.nan)
    maximum = np.where(np.isfinite(maximum), maximum, np.nan)

    mean_view = downsample(mean, max_points)
    std_view = downsample(std, max_points)
    zero_view = downsample(zero_ratio, max_points)
    min_view = downsample(minimum, max_points)
    max_view = downsample(maximum, max_points)
    if any(view["downsampled"] for view in (mean_view, std_view, zero_view)):
        warnings.append(
            "per-dimension arrays were bucket-averaged to at most %d points per "
            "series; see per_dimension.downsampled in the report" % max_points
        )

    top_variance = []
    for index in top_indices(std, 10):
        top_variance.append(
            {
                "dimension": int(index),
                "std": as_finite_float(std[index]),
                "mean": as_finite_float(mean[index]),
                "zero_ratio": as_finite_float(zero_ratio[index]),
            }
        )
    top_zero = []
    for index in top_indices(zero_ratio, 10):
        top_zero.append(
            {
                "dimension": int(index),
                "zero_ratio": as_finite_float(zero_ratio[index]),
                "mean": as_finite_float(mean[index]),
            }
        )

    return metric_entry(
        DEFINITIONS["dimensions"],
        "full",
        None,
        dim=int(mean.size),
        summary={
            "mean": summarize(mean),
            "std": summarize(std),
            "zero_ratio": summarize(zero_ratio),
        },
        top_variance_dimensions=top_variance,
        top_zero_dimensions=top_zero,
        per_dimension={
            "mean": mean_view,
            "std": std_view,
            "zero_ratio": zero_view,
            "min": min_view,
            "max": max_view,
        },
    )


def _dense_sparsity_metric(
    total_values: int, zero_values: int, all_zero_vectors: int, count: int
) -> Dict[str, Any]:
    nonzero = max(0, total_values - zero_values)
    density = nonzero / float(total_values) if total_values else None
    return metric_entry(
        DEFINITIONS["sparsity"],
        "full",
        None,
        representation="dense",
        count=count,
        total_values=int(total_values),
        zero_values=int(zero_values),
        non_zero_values=int(nonzero),
        zero_ratio=as_finite_float(zero_values / float(total_values) if total_values else None),
        density=as_finite_float(density),
        all_zero_vectors=int(all_zero_vectors),
        all_zero_vector_ratio=as_finite_float(
            all_zero_vectors / float(count) if count else None
        ),
    )


def _sparse_sparsity_metric(
    count: int,
    dim: int,
    stored_entries: int,
    nonzero_entries: int,
    nnz_per_vector: np.ndarray,
    all_zero_vectors: int,
) -> Dict[str, Any]:
    cells = count * dim if dim else None
    density = nonzero_entries / float(cells) if cells else None
    return metric_entry(
        DEFINITIONS["sparsity"],
        "full",
        None,
        representation="sparse",
        count=int(count),
        dim=int(dim),
        cells=cells,
        stored_entries=int(stored_entries),
        non_zero_values=int(nonzero_entries),
        explicit_zero_entries=int(max(0, stored_entries - nonzero_entries)),
        zero_ratio=as_finite_float(1.0 - density) if density is not None else None,
        density=as_finite_float(density),
        nnz_per_vector=summarize(nnz_per_vector),
        nnz_histogram=histogram(nnz_per_vector, bins=min(HISTOGRAM_BINS, 20)),
        all_zero_vectors=int(all_zero_vectors),
        all_zero_vector_ratio=as_finite_float(
            all_zero_vectors / float(count) if count else None
        ),
    )


def _duplicates_metric(
    hashes: Counter,
    count: int,
    pair_values: np.ndarray,
    epsilon: float,
) -> Dict[str, Any]:
    duplicate_vectors = sum(value - 1 for value in hashes.values() if value > 1)
    duplicate_groups = sum(1 for value in hashes.values() if value > 1)
    max_group = max(hashes.values()) if hashes else 0
    exact = {
        "definition": "Vectors whose complete byte representation is identical.",
        "scope": "full",
        "sample_size": None,
        "count": int(count),
        "unique_vectors": int(len(hashes)),
        "duplicate_vectors": int(duplicate_vectors),
        "duplicate_ratio": as_finite_float(
            duplicate_vectors / float(count) if count else None
        ),
        "duplicate_groups": int(duplicate_groups),
        "max_group_size": int(max_group),
    }
    near = _near_duplicate_summary(pair_values, epsilon)
    return metric_entry(
        DEFINITIONS["duplicates"],
        "mixed",
        None,
        exact=exact,
        near_duplicate=near,
        scope_note=(
            "exact duplicate detection covers the full split; the near-duplicate "
            "indicator covers the bounded pairwise sample"
        ),
    )


def _near_duplicate_summary(pair_values: np.ndarray, epsilon: float) -> Dict[str, Any]:
    if pair_values.size == 0:
        return {
            "available": False,
            "reason": "fewer than two sampled vectors",
            "scope": "sample",
            "sample_size": 0,
            "pair_count": 0,
        }
    finite = pair_values[np.isfinite(pair_values)]
    if finite.size == 0:
        return {
            "available": False,
            "reason": "no finite sampled pairwise distances",
            "scope": "sample",
            "sample_size": 0,
            "pair_count": int(pair_values.size),
        }
    median = float(np.median(finite))
    threshold = max(0.0, median * float(epsilon))
    near_count = int(np.count_nonzero(finite <= threshold))
    return {
        "available": True,
        "definition": (
            "Sampled pairs whose distance is at most epsilon times the sampled "
            "pairwise-distance median."
        ),
        "scope": "sample",
        "sample_size": int(_sample_size_from_pair_count(pair_values.size)),
        "pair_count": int(finite.size),
        "relative_epsilon": as_finite_float(epsilon),
        "median_pairwise_distance": as_finite_float(median),
        "threshold": as_finite_float(threshold),
        "near_duplicate_pairs": near_count,
        "near_duplicate_ratio": as_finite_float(near_count / float(finite.size)),
    }


def _sample_size_from_pair_count(pair_count: int) -> int:
    # pair_count = M * (M - 1) / 2
    if pair_count <= 0:
        return 0
    return int((1 + math.isqrt(1 + 8 * pair_count)) // 2)


def _sampled_metrics(
    *,
    pair_matrix: np.ndarray,
    sample_matrix: Optional[np.ndarray],
    pair_values: np.ndarray,
    metric: str,
    config: ProfilerConfig,
    rng: np.random.Generator,
    warnings: List[str],
    sparse_pairwise: bool = False,
) -> Dict[str, Any]:
    sample_size = int(_sample_size_from_pair_count(pair_values.size))
    pair_count = int(pair_values.size)
    pairwise = metric_entry(
        DEFINITIONS["pairwise_distances"],
        "sample",
        sample_size,
        metric=metric,
        pair_count=pair_count,
        requested_pairwise_sample_size=int(config.pairwise_sample_size),
        stats=summarize(pair_values),
        histogram=histogram(pair_values),
        matrix_is_bounded=True,
        dense_sample_materialized=bool(sample_matrix is not None),
        sparse_sparse_distance_loops=bool(sparse_pairwise),
    )
    nearest = _nearest_neighbor_metric(pair_matrix)
    anisotropy = _anisotropy_metric(sample_matrix, pair_matrix, metric, warnings)
    clustering = _clustering_metric(
        sample_matrix, config.clusters, rng, warnings
    )
    return {
        "pairwise_distances": pairwise,
        "nearest_neighbor_distances": nearest,
        "anisotropy": anisotropy,
        "clustering": clustering,
    }


def _nearest_neighbor_metric(pair_matrix: np.ndarray) -> Dict[str, Any]:
    sample_size = int(pair_matrix.shape[0])
    if sample_size < 2:
        return metric_entry(
            DEFINITIONS["nearest_neighbor_distances"],
            "sample",
            sample_size,
            available=False,
            reason="fewer than two sampled vectors",
        )
    matrix = np.array(pair_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(matrix, np.inf)
    values = np.min(matrix, axis=1)
    values = values[np.isfinite(values)]
    return metric_entry(
        DEFINITIONS["nearest_neighbor_distances"],
        "sample",
        sample_size,
        available=True,
        stats=summarize(values),
        histogram=histogram(values),
    )


def _anisotropy_metric(
    sample_matrix: Optional[np.ndarray],
    pair_matrix: np.ndarray,
    metric: str,
    warnings: List[str],
) -> Dict[str, Any]:
    sample_size = (
        int(sample_matrix.shape[0])
        if sample_matrix is not None
        else int(pair_matrix.shape[0])
    )
    if sample_matrix is None:
        return metric_entry(
            DEFINITIONS["anisotropy"],
            "sample",
            sample_size,
            available=False,
            reason=(
                "the sampled vectors could not be materialized as a dense matrix "
                "within the dimensionality/memory guard"
            ),
        )
    if sample_size < 3 or sample_matrix.shape[1] == 0:
        return metric_entry(
            DEFINITIONS["anisotropy"],
            "sample",
            sample_size,
            available=False,
            reason="needs at least three sampled vectors with non-zero dimensionality",
        )
    result = _pca_statistics(sample_matrix)
    if result is None:
        return metric_entry(
            DEFINITIONS["anisotropy"],
            "sample",
            sample_size,
            available=False,
            reason="sample contains no finite variation",
        )
    eigenvalues, projection, explained = result
    total = float(np.sum(eigenvalues))
    if total > 0:
        shares = eigenvalues / total
        top1 = float(shares[0])
        top10 = float(np.sum(shares[:10]))
        entropy = -float(np.sum(shares[shares > 0] * np.log(shares[shares > 0])))
        effective_rank = float(math.exp(entropy))
    else:
        top1 = top10 = effective_rank = None
    mean_cosine = _mean_pairwise_cosine(sample_matrix)
    return metric_entry(
        DEFINITIONS["anisotropy"],
        "sample",
        sample_size,
        available=True,
        method="eigendecomposition of centered sample covariance",
        top1_eigenvalue_share=as_finite_float(top1),
        top10_eigenvalue_share=as_finite_float(top10),
        effective_rank=as_finite_float(effective_rank),
        mean_pairwise_cosine=as_finite_float(mean_cosine),
        explained_variance_ratio=[as_finite_float(value) for value in explained[:10]],
        interpretation=(
            "top1_eigenvalue_share near 1 indicates a dominant direction; "
            "effective_rank near 1 indicates nearly one-dimensional variation"
        ),
    )


def _clustering_metric(
    sample_matrix: Optional[np.ndarray],
    clusters: int,
    rng: np.random.Generator,
    warnings: List[str],
) -> Dict[str, Any]:
    sample_size = 0 if sample_matrix is None else int(sample_matrix.shape[0])
    if sample_matrix is None:
        return metric_entry(
            DEFINITIONS["clustering"],
            "sample",
            sample_size,
            available=False,
            reason="sampled vectors could not be materialized as a dense matrix",
        )
    if sample_matrix.shape[0] < 3 or sample_matrix.shape[1] == 0:
        return metric_entry(
            DEFINITIONS["clustering"],
            "sample",
            sample_size,
            available=False,
            reason="needs at least three sampled vectors with non-zero dimensionality",
        )
    result = _pca_statistics(sample_matrix)
    if result is None:
        return metric_entry(
            DEFINITIONS["clustering"],
            "sample",
            sample_size,
            available=False,
            reason="sample contains no finite variation",
        )
    _, projection, explained = result
    output: Dict[str, Any] = {
        "available": True,
        "method": "deterministic PCA-2 projection of the seeded sample",
        "projection_points": [
            [as_finite_float(point[0]), as_finite_float(point[1])]
            for point in projection
        ],
        "explained_variance_ratio_pc1": as_finite_float(
            explained[0] if explained.size else None
        ),
        "explained_variance_ratio_pc2": as_finite_float(
            explained[1] if explained.size > 1 else None
        ),
    }
    if clusters <= 0:
        output["clustering_available"] = False
        output["reason"] = "disabled (pass --clusters K to enable k-means)"
        return metric_entry(
            DEFINITIONS["clustering"], "sample", sample_size, **output
        )
    labels, sizes, inertia, iterations = _kmeans(projection, clusters, rng)
    output.update(
        {
            "clustering_available": True,
            "n_clusters": int(len(sizes)),
            "cluster_sizes": [int(value) for value in sizes],
            "inertia": as_finite_float(inertia),
            "iterations": int(iterations),
            "labels": [int(value) for value in labels],
        }
    )
    return metric_entry(DEFINITIONS["clustering"], "sample", sample_size, **output)


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------
def _pairwise_dense(matrix: np.ndarray, metric: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if metric == "euclidean":
        squared_norms = np.einsum("ij,ij->i", matrix, matrix)
        squared = (
            squared_norms[:, None] + squared_norms[None, :] - 2.0 * (matrix @ matrix.T)
        )
        np.maximum(squared, 0.0, out=squared)
        return np.sqrt(squared)
    if metric == "ip":
        return 1.0 - matrix @ matrix.T
    if metric == "angular":
        norms = np.linalg.norm(matrix, axis=1)
        safe = np.where(norms > 0, norms, 1.0)
        normalized = matrix / safe[:, None]
        similarity = normalized @ normalized.T
        np.clip(similarity, -1.0, 1.0, out=similarity)
        distances = 1.0 - similarity
        zero_norms = norms == 0
        if np.any(zero_norms):
            distances[zero_norms, :] = 1.0
            distances[:, zero_norms] = 1.0
            distances[np.ix_(zero_norms, zero_norms)] = 0.0
        return distances
    raise ValueError("unsupported metric %r" % metric)


def _pairwise_sparse(
    records: Sequence[Tuple[int, np.ndarray, np.ndarray]], metric: str
) -> np.ndarray:
    size = len(records)
    matrix = np.zeros((size, size), dtype=np.float64)
    maps: List[Dict[int, float]] = []
    norms = np.zeros(size, dtype=np.float64)
    for position, (_, ids, values) in enumerate(records):
        mapping = {int(key): float(value) for key, value in zip(ids, values) if np.isfinite(value)}
        maps.append(mapping)
        norms[position] = math.sqrt(sum(value * value for value in mapping.values()))
    for i in range(size):
        for j in range(i + 1, size):
            left = maps[i]
            right = maps[j]
            if len(left) > len(right):
                left, right = right, left
            dot = 0.0
            for key, value in left.items():
                other = right.get(key)
                if other is not None:
                    dot += value * other
            if metric == "ip":
                distance = 1.0 - dot
            elif metric == "euclidean":
                squared = norms[i] * norms[i] + norms[j] * norms[j] - 2.0 * dot
                distance = math.sqrt(max(0.0, squared))
            elif metric == "angular":
                if norms[i] == 0.0 and norms[j] == 0.0:
                    distance = 0.0
                elif norms[i] == 0.0 or norms[j] == 0.0:
                    distance = 1.0
                else:
                    cosine = max(-1.0, min(1.0, dot / (norms[i] * norms[j])))
                    distance = 1.0 - cosine
            else:
                raise ValueError("unsupported metric %r" % metric)
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _sparse_sample_to_dense(
    records: Sequence[Tuple[int, np.ndarray, np.ndarray]], dim: int
) -> Optional[np.ndarray]:
    if dim <= 0 or not records:
        return None
    rows = len(records)
    if rows * dim * 8 > _MAX_ANISOTROPY_BYTES or dim > _MAX_ANISOTROPY_DIM:
        return None
    matrix = np.zeros((rows, dim), dtype=np.float64)
    for position, (_, ids, values) in enumerate(records):
        if ids.size:
            matrix[position, ids.astype(np.int64, copy=False)] = values
    return np.where(np.isfinite(matrix), matrix, 0.0)


# ---------------------------------------------------------------------------
# PCA / clustering helpers
# ---------------------------------------------------------------------------
def _pca_statistics(
    matrix: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 3 or matrix.shape[1] == 0:
        return None
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    if not np.any(np.isfinite(centered)) or not np.any(centered != 0.0):
        return None
    rows, dim = centered.shape
    if dim <= 4096:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (singular * singular) / float(max(rows - 1, 1))
        total = float(np.sum(eigenvalues))
        explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        components = vt[:2] if vt.shape[0] >= 2 else vt
        projection = centered @ components.T
    else:
        gram = centered @ centered.T
        eigenvalues_raw, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues = np.maximum(eigenvalues_raw[order], 0.0) / float(max(rows - 1, 1))
        total = float(np.sum(eigenvalues))
        explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        scores = eigenvectors[:, order[:2]] * np.sqrt(np.maximum(eigenvalues_raw[order[:2]], 0.0))
        projection = scores
    if projection.shape[1] < 2:
        projection = np.pad(projection, ((0, 0), (0, 2 - projection.shape[1])))
    if projection.shape[0] == 0:
        return None
    return eigenvalues, projection, explained


def _mean_pairwise_cosine(matrix: np.ndarray) -> Optional[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return None
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    normalized = matrix / safe[:, None]
    gram = normalized @ normalized.T
    upper = gram[np.triu_indices(gram.shape[0], k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return None
    return float(np.mean(np.clip(upper, -1.0, 1.0)))


def _kmeans(
    points: np.ndarray, k: int, rng: np.random.Generator, max_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    points = np.asarray(points, dtype=np.float64)
    rows = points.shape[0]
    k = max(1, min(int(k), rows))
    initial = sample_indices(rows, k, rng)
    centers = points[initial].copy()
    labels = np.zeros(rows, dtype=np.int64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and iterations > 1:
            labels = new_labels
            break
        labels = new_labels
        for cluster in range(k):
            members = points[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centers[cluster] = points[farthest]
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(np.min(distances, axis=1) ** 2))
    sizes = np.bincount(labels, minlength=k)
    return labels, sizes, inertia, iterations


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def _digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=16).digest()
