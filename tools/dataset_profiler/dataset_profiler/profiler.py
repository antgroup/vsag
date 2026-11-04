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
_MAX_SPARSE_STATS_DIMENSION = 10_000_000
_PCA_SVD_MAX_DIM = 4096
# Dense streaming blocks are sized by their working bytes, not by a fixed row
# count: every row keeps the raw block, a float64 copy, a finite mask, and
# several float64 temporaries alive at the same time.
_DENSE_BLOCK_BYTES = 128 * 1024 * 1024
_DENSE_BLOCK_BYTES_PER_VALUE = 40
_DENSE_BLOCK_MAX_ROWS = 65536
# Relative tolerance under which the expanded Euclidean formula (||a||²+||b||²-2a·b)
# has lost every significant digit; such pairs are recomputed from differences.
_EUCLIDEAN_GRAM_REL_TOL = 1e-8
# Normalization and dot products can leave identical angular vectors a few ulps
# from zero.  Canonicalize that numerical residue before duplicate thresholds.
_ANGULAR_ZERO_TOL = 32 * np.finfo(np.float64).eps
# Row budget for the streaming mean-cosine reduction (normalized blocks only).
_MEAN_COSINE_BLOCK_ROWS = 8192
_INVALID_SAMPLE_WARNING = (
    "sampled vectors contain NaN/Inf entries; sampled metrics replace them with "
    "0.0, which can distort pairwise, nearest-neighbor, anisotropy, and clustering "
    "results"
)


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
        dataset_count = int(metadata.count or 0)
        dataset_limited_sample = min(requested_sample, dataset_count)
        effective_sample = dataset_limited_sample
        if metadata.vector_type == "dense":
            effective_sample = _dense_sample_size(
                dataset_count, int(metadata.dim or 0), requested_sample
            )
        effective_pairwise = min(requested_pairwise, effective_sample)
        if dataset_limited_sample < requested_sample:
            warnings.append(
                "sample_size reduced from %d to %d to stay within the dataset size"
                % (requested_sample, dataset_limited_sample)
            )
        if effective_sample < dataset_limited_sample:
            warnings.append(
                "sample_size reduced from %d to %d to stay within the dense "
                "sample memory budget" % (dataset_limited_sample, effective_sample)
            )
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
                "uniformly random subset selected by a seeded permutation of the "
                "sampled rows"
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
def _dense_block_rows(dim: int) -> int:
    """Return a dense block row count whose working buffers fit the byte budget.

    The streaming dense pass converts each block to float64 and keeps several
    float64/bool temporaries alive simultaneously, so a fixed row count would
    retain gigabytes for wide vectors.  Row counts are derived from
    ``_DENSE_BLOCK_BYTES`` instead.
    """
    if dim <= 0:
        return _DENSE_BLOCK_MAX_ROWS
    rows = _DENSE_BLOCK_BYTES // (int(dim) * _DENSE_BLOCK_BYTES_PER_VALUE)
    return int(max(1, min(_DENSE_BLOCK_MAX_ROWS, rows)))


def _dense_sample_size(count: int, dim: int, requested: int) -> int:
    """Cap the dense float64 sample by its materialized byte size."""
    dataset_limited = min(int(requested), int(count))
    if dim <= 0:
        return int(dataset_limited)
    max_rows = _MAX_ANISOTROPY_BYTES // (int(dim) * 8)
    return int(min(dataset_limited, max_rows))


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
    sample_size = _dense_sample_size(count, dim, config.sample_size)
    sample_idx = sample_indices(count, sample_size, rng)
    pair_n = min(config.pairwise_sample_size, sample_idx.size)
    # Sorting only restores row order after selection; the subset itself comes
    # from a seeded permutation and is not biased toward low dataset indices.
    pair_positions = np.sort(rng.permutation(sample_idx.size)[:pair_n])

    dim_count = np.zeros(dim, dtype=np.float64)
    dim_mean = np.zeros(dim, dtype=np.float64)
    dim_m2 = np.zeros(dim, dtype=np.float64)
    dim_zero = np.zeros(dim, dtype=np.float64)
    dim_min = np.full(dim, np.inf, dtype=np.float64)
    dim_max = np.full(dim, -np.inf, dtype=np.float64)

    hashes: Counter = Counter()
    norms_parts: List[np.ndarray] = []
    total_values = 0
    zero_values = 0
    nonzero_values = 0
    invalid_values = 0
    vectors_with_invalid = 0
    all_zero_vectors = 0

    for _, block in dataset.iter_dense_blocks(block_rows=_dense_block_rows(dim)):
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
        block_zero_values = int(np.count_nonzero((raw == 0) & finite))
        total_values += int(raw.size)
        zero_values += block_zero_values
        nonzero_values += int(raw.size) - invalid_in_block - block_zero_values

        norms = np.sqrt(np.einsum("ij,ij->i", valid, valid))
        norms_parts.append(norms)
        all_zero_vectors += int(np.count_nonzero(norms == 0))

        # Merge each block with the running per-coordinate Welford state.  In
        # particular, do not subtract E[x]^2 from E[x^2]: that loses small
        # variations around large FLOAT32 magnitudes.
        block_count = finite.sum(axis=0).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            block_mean = np.where(
                block_count > 0, valid.sum(axis=0) / block_count, 0.0
            )
        centered = np.where(finite, valid - block_mean, 0.0)
        block_m2 = np.einsum("ij,ij->j", centered, centered)
        combined_count = dim_count + block_count
        delta = block_mean - dim_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(
                combined_count > 0, block_count / combined_count, 0.0
            )
            correction = np.where(
                combined_count > 0,
                delta * delta * dim_count * weight,
                0.0,
            )
        dim_mean += delta * weight
        dim_m2 += block_m2 + correction
        dim_count = combined_count
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
    finite_sample = np.isfinite(sample_matrix)
    if not bool(finite_sample.all()):
        warnings.append(_INVALID_SAMPLE_WARNING)
        # Replace invalid entries in place to avoid a second full sample-sized
        # allocation; full-dataset numeric summaries continue to exclude them.
        sample_matrix[~finite_sample] = 0.0
    pair_matrix = sample_matrix[pair_positions]
    pair_distances = _pairwise_dense(pair_matrix, metric)
    pair_values = _upper_triangle(pair_distances)
    duplicate_pair_values = pair_values
    duplicate_metric = metric
    if metric == "ip":
        # IP distance (1 - dot) is negative for unnormalized duplicates and is
        # therefore unsuitable for a non-negative near-duplicate threshold.
        duplicate_pair_values = _upper_triangle(_pairwise_dense(pair_matrix, "euclidean"))
        duplicate_metric = "euclidean"

    metrics: Dict[str, Any] = {
        "invalid_values": _invalid_metric(
            invalid_values, vectors_with_invalid, total_values, count
        ),
        "norms": _norms_metric(norms),
        "outliers": _outliers_metric(norms),
        "dimensions": _dimensions_metric(
            dim_count,
            dim_mean,
            dim_m2,
            dim_zero,
            dim_min,
            dim_max,
            config.max_dimension_points,
            warnings,
        ),
        "sparsity": _dense_sparsity_metric(
            total_values, zero_values, nonzero_values, all_zero_vectors, count
        ),
        "duplicates": _duplicates_metric(
            hashes,
            count,
            duplicate_pair_values,
            config.near_duplicate_relative_epsilon,
            duplicate_metric,
            int(pair_matrix.shape[0]),
        ),
    }
    metrics.update(
        _sampled_metrics(
            pair_matrix=pair_distances,
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
    pair_positions = np.sort(rng.permutation(sample_idx.size)[:pair_n])
    sample_set = set(int(index) for index in sample_idx)
    pair_sample_set = set(int(sample_idx[position]) for position in pair_positions)

    accumulator: Optional[_DimAccumulator] = _DimAccumulator()
    dimension_stats_reason: Optional[str] = None
    dim = 0
    hashes: Counter = Counter()
    # The record count is known before this pass (a no-offsets stream is counted
    # above), so avoid retaining one tiny NumPy array per sparse vector.
    norms = np.empty(count, dtype=np.float64)
    nnz = np.empty(count, dtype=np.int64)
    stored_entries = 0
    nonzero_entries = 0
    explicit_zero_entries = 0
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
        if ids.size > 1 and np.any(ids[1:] == ids[:-1]):
            # Pairwise/PCA materialization keeps one value per feature, while
            # duplicate observations would corrupt full per-dimension means.
            # Reject the ambiguous encoding instead of mixing both semantics.
            raise ValueError("sparse record %d contains repeated feature IDs" % index)
        length = int(ids.size)
        stored_entries += length
        finite = np.isfinite(values)
        valid = np.where(finite, values, 0).astype(np.float64, copy=False)
        invalid_values += int((~finite).sum())
        if not bool(finite.all()):
            vectors_with_invalid += 1
        nonzero = int(np.count_nonzero(valid))
        nonzero_entries += nonzero
        explicit_zero_entries += int(np.count_nonzero((values == 0) & finite))
        norm = float(math.sqrt(float(np.dot(valid, valid))))
        norms[index] = norm
        nnz[index] = nonzero
        if norm == 0.0:
            all_zero_vectors += 1
        if ids.size:
            dim = max(dim, int(ids[-1]) + 1)
        if dim > _MAX_SPARSE_STATS_DIMENSION and accumulator is not None:
            dimension_stats_reason = (
                "sparse dimensionality %d exceeds the per-dimension statistics "
                "limit of %d; other metrics are still reported"
                % (dim, _MAX_SPARSE_STATS_DIMENSION)
            )
            warnings.append(dimension_stats_reason)
            # Release the bounded accumulators before continuing the full pass.
            accumulator = None
        if accumulator is not None:
            accumulator.add(ids, values, finite)
        hashes[_digest(ids.astype("<u4", copy=False).tobytes() + values.astype("<f4", copy=False).tobytes())] += 1
        if index in sample_set:
            sample_records.append((index, ids.copy(), values.copy()))

    if len(sample_records) != len(sample_set):
        raise ValueError("sparse sampling failed: not all sampled records were found")
    sample_records.sort(key=lambda item: item[0])
    pair_records = [
        record for record in sample_records if record[0] in pair_sample_set
    ]
    metadata.dim = dim
    if dim == 0:
        warnings.append(
            "all sparse vectors are empty; dimensionality, density, and "
            "per-dimension statistics are unavailable"
        )

    if accumulator is not None:
        dim_count, dim_mean, dim_m2, dim_zero, dim_min, dim_max = (
            accumulator.arrays(count)
        )
        dimensions = _dimensions_metric(
            dim_count,
            dim_mean,
            dim_m2,
            dim_zero,
            dim_min,
            dim_max,
            config.max_dimension_points,
            warnings,
        )
    else:
        dimensions = metric_entry(
            DEFINITIONS["dimensions"],
            "full",
            None,
            available=False,
            reason=dimension_stats_reason,
            dim=int(dim),
        )
    if any(
        not bool(np.isfinite(values).all()) for _, _, values in sample_records
    ):
        warnings.append(_INVALID_SAMPLE_WARNING)
    pair_matrix = _pairwise_sparse(pair_records, metric)
    pair_values = _upper_triangle(pair_matrix)
    duplicate_pair_values = pair_values
    duplicate_metric = metric
    if metric == "ip":
        duplicate_pair_values = _upper_triangle(
            _pairwise_sparse(pair_records, "euclidean")
        )
        duplicate_metric = "euclidean"
    sample_matrix = _sparse_sample_to_dense(sample_records, dim)

    metrics: Dict[str, Any] = {
        "invalid_values": _invalid_metric(
            invalid_values, vectors_with_invalid, count * dim if dim else None, count
        ),
        "norms": _norms_metric(norms),
        "outliers": _outliers_metric(norms),
        "dimensions": dimensions,
        "sparsity": _sparse_sparsity_metric(
            count,
            dim,
            stored_entries,
            nonzero_entries,
            invalid_values,
            explicit_zero_entries,
            nnz,
            all_zero_vectors,
        ),
        "duplicates": _duplicates_metric(
            hashes,
            count,
            duplicate_pair_values,
            config.near_duplicate_relative_epsilon,
            duplicate_metric,
            int(pair_matrix.shape[0]),
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
    """Bounded growable Welford accumulators for sparse dimension data."""

    def __init__(self) -> None:
        self._max_index = -1
        self.count = np.zeros(0, dtype=np.float64)
        self.invalid = np.zeros(0, dtype=np.float64)
        self.mean = np.zeros(0, dtype=np.float64)
        self.m2 = np.zeros(0, dtype=np.float64)
        self.zero = np.zeros(0, dtype=np.float64)
        self.min = np.full(0, np.inf, dtype=np.float64)
        self.max = np.full(0, -np.inf, dtype=np.float64)

    @property
    def dim(self) -> int:
        return self._max_index + 1

    def _ensure(self, size: int) -> None:
        if size <= self.count.size:
            return
        if size > _MAX_SPARSE_STATS_DIMENSION:
            # The profiling loop normally disables dimension metrics before
            # reaching this guard.  Keep the accumulator itself safe as well.
            raise ValueError("sparse dimension exceeds per-dimension statistics limit")
        new_size = min(
            max(size, max(16, self.count.size * 2)), _MAX_SPARSE_STATS_DIMENSION
        )
        pad = new_size - self.count.size
        self.count = np.concatenate([self.count, np.zeros(pad)])
        self.invalid = np.concatenate([self.invalid, np.zeros(pad)])
        self.mean = np.concatenate([self.mean, np.zeros(pad)])
        self.m2 = np.concatenate([self.m2, np.zeros(pad)])
        self.zero = np.concatenate([self.zero, np.zeros(pad)])
        self.min = np.concatenate([self.min, np.full(pad, np.inf)])
        self.max = np.concatenate([self.max, np.full(pad, -np.inf)])

    def add(self, ids: np.ndarray, values: np.ndarray, finite: np.ndarray) -> None:
        if ids.size == 0:
            return
        self._max_index = max(self._max_index, int(ids.max()))
        self._ensure(self._max_index + 1)
        all_indices = ids.astype(np.int64, copy=False)
        if not bool(finite.all()):
            np.add.at(self.invalid, all_indices[~finite], 1.0)
        if not np.any(finite):
            return
        index = all_indices[finite]
        vals = values[finite].astype(np.float64, copy=False)

        # Merge one stable per-record batch for each feature.  np.unique also
        # makes repeated feature IDs deterministic without scalar Welford loops.
        unique, inverse = np.unique(index, return_inverse=True)
        group_count = np.bincount(inverse).astype(np.float64)
        group_sum = np.bincount(inverse, weights=vals)
        group_mean = group_sum / group_count
        centered = vals - group_mean[inverse]
        group_m2 = np.bincount(inverse, weights=centered * centered)
        old_count = self.count[unique]
        old_mean = self.mean[unique]
        combined_count = old_count + group_count
        delta = group_mean - old_mean
        weight = group_count / combined_count
        self.mean[unique] = old_mean + delta * weight
        self.m2[unique] = (
            self.m2[unique] + group_m2 + delta * delta * old_count * weight
        )
        self.count[unique] = combined_count

        zero = vals == 0
        if np.any(zero):
            np.add.at(self.zero, index[zero], 1.0)
        np.minimum.at(self.min, index, vals)
        np.maximum.at(self.max, index, vals)

    def arrays(self, vector_count: int):
        """Merge implicit zeros stably while excluding stored invalid values."""
        dim = self.dim
        finite_count = self.count[:dim]
        invalid_count = self.invalid[:dim]
        count = float(vector_count) - invalid_count
        implicit_count = np.maximum(count - finite_count, 0.0)
        mean = self.mean[:dim].copy()
        m2 = self.m2[:dim].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_weight = np.where(count > 0, finite_count / count, 0.0)
            zero_correction = np.where(
                count > 0,
                mean * mean * finite_count * (implicit_count / count),
                0.0,
            )
        mean *= mean_weight
        m2 += zero_correction
        zero = self.zero[:dim] + implicit_count
        implicit_zero = implicit_count > 0
        minimum = self.min[:dim].copy()
        maximum = self.max[:dim].copy()
        minimum[implicit_zero] = np.minimum(minimum[implicit_zero], 0.0)
        maximum[implicit_zero] = np.maximum(maximum[implicit_zero], 0.0)
        return count, mean, m2, zero, minimum, maximum


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
        elif np.all(values == median):
            mask = np.zeros(values.shape, dtype=bool)
            method = "all norms identical; no outlier can be distinguished"
            threshold = 0.0
        else:
            # With a repeated central value, both robust spreads can be zero
            # even though extreme values exist (for example 99 ones and 100).
            mask = values != median
            method = "median fallback (MAD and IQR are zero)"
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
    mean_value: np.ndarray,
    m2: np.ndarray,
    zero: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    max_points: int,
    warnings: List[str],
) -> Dict[str, Any]:
    mean = np.asarray(mean_value, dtype=np.float64).copy()
    mean[count <= 0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(count > 0, m2 / count, np.nan)
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
    total_values: int,
    zero_values: int,
    nonzero_values: int,
    all_zero_vectors: int,
    count: int,
) -> Dict[str, Any]:
    density = nonzero_values / float(total_values) if total_values else None
    return metric_entry(
        DEFINITIONS["sparsity"],
        "full",
        None,
        representation="dense",
        count=count,
        total_values=int(total_values),
        zero_values=int(zero_values),
        non_zero_values=int(nonzero_values),
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
    invalid_values: int,
    explicit_zero_entries: int,
    nnz_per_vector: np.ndarray,
    all_zero_vectors: int,
) -> Dict[str, Any]:
    cells = count * dim if dim else None
    zero_values = (
        cells - nonzero_entries - invalid_values if cells is not None else None
    )
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
        zero_values=int(zero_values) if zero_values is not None else None,
        non_zero_values=int(nonzero_entries),
        explicit_zero_entries=int(explicit_zero_entries),
        zero_ratio=as_finite_float(
            zero_values / float(cells) if zero_values is not None and cells else None
        ),
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
    near_metric: str,
    near_sample_size: int,
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
    near = _near_duplicate_summary(
        pair_values, epsilon, near_metric, near_sample_size
    )
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


def _near_duplicate_summary(
    pair_values: np.ndarray, epsilon: float, metric: str, sample_size: int
) -> Dict[str, Any]:
    if pair_values.size == 0:
        return {
            "available": False,
            "reason": "fewer than two sampled vectors",
            "scope": "sample",
            "sample_size": int(sample_size),
            "pair_count": 0,
        }
    finite = pair_values[np.isfinite(pair_values)]
    if finite.size == 0:
        return {
            "available": False,
            "reason": "no finite sampled pairwise distances",
            "scope": "sample",
            "sample_size": int(sample_size),
            "pair_count": int(pair_values.size),
        }
    median = float(np.median(finite))
    threshold = max(0.0, median * float(epsilon))
    near_count = int(np.count_nonzero(finite <= threshold))
    return {
        "available": True,
        "definition": (
            "Sampled pairs whose duplicate-detection distance is at most epsilon "
            "times the sampled pairwise-distance median."
        ),
        "metric": metric,
        "scope": "sample",
        "sample_size": int(sample_size),
        "pair_count": int(finite.size),
        "relative_epsilon": as_finite_float(epsilon),
        "median_pairwise_distance": as_finite_float(median),
        "threshold": as_finite_float(threshold),
        "near_duplicate_pairs": near_count,
        "near_duplicate_ratio": as_finite_float(near_count / float(finite.size)),
    }


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
    sample_size = int(pair_matrix.shape[0])
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
    pca_result = (
        _pca_statistics(sample_matrix) if sample_matrix is not None else None
    )
    anisotropy = _anisotropy_metric(
        sample_matrix, pair_matrix, metric, warnings, pca_result
    )
    clustering = _clustering_metric(
        sample_matrix, config.clusters, rng, warnings, pca_result
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
    pca_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
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
    result = pca_result
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
    pca_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
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
    result = pca_result
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
        return _dense_euclidean(matrix)
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
        distances[np.abs(distances) <= _ANGULAR_ZERO_TOL] = 0.0
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
                # Direct coordinate differences avoid the cancellation that the
                # expanded norm/dot form suffers for large-magnitude vectors.
                squared = 0.0
                for key in left.keys() | right.keys():
                    delta = left.get(key, 0.0) - right.get(key, 0.0)
                    squared += delta * delta
                distance = math.sqrt(squared)
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
            if metric == "angular" and abs(distance) <= _ANGULAR_ZERO_TOL:
                distance = 0.0
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def _dense_euclidean(matrix: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance matrix without catastrophic cancellation.

    The fast expanded form is used first.  When the norms of two rows dwarf
    their distance (for example rows near 1e10 whose differences are 1), that
    expansion rounds to zero, so every pair touching an unstable row is
    recomputed from direct coordinate differences.
    """
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (
        matrix @ matrix.T
    )
    scale = squared_norms[:, None] + squared_norms[None, :]
    unstable = (scale > 0) & (squared <= _EUCLIDEAN_GRAM_REL_TOL * scale)
    # A zero self-distance always satisfies the relative test.  It needs no
    # correction and must not send every ordinary row through the O(M*D)
    # direct-difference fallback.
    np.fill_diagonal(unstable, False)
    if np.any(unstable):
        for row in np.unique(np.nonzero(unstable)[0]):
            difference = matrix[row] - matrix
            direct = np.einsum("ij,ij->i", difference, difference)
            squared[row, :] = direct
            squared[:, row] = direct
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


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
    if dim <= _PCA_SVD_MAX_DIM:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (singular * singular) / float(max(rows - 1, 1))
        total = float(np.sum(eigenvalues))
        explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        components = vt[:2] if vt.shape[0] >= 2 else vt
        projection = centered @ components.T
    elif rows <= dim:
        # For tall-enough wide inputs the row Gram matrix is the smaller square
        # decomposition and avoids a dim-by-dim covariance allocation.
        gram = centered @ centered.T
        eigenvalues_raw, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues = np.maximum(eigenvalues_raw[order], 0.0) / float(max(rows - 1, 1))
        total = float(np.sum(eigenvalues))
        explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        scores = eigenvectors[:, order[:2]] * np.sqrt(np.maximum(eigenvalues_raw[order[:2]], 0.0))
        projection = scores
    else:
        # When rows outnumber a wide feature dimension, the feature covariance
        # is smaller than the rows-by-rows Gram matrix.
        covariance = centered.T @ centered
        eigenvalues_raw, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues = np.maximum(eigenvalues_raw[order], 0.0) / float(max(rows - 1, 1))
        total = float(np.sum(eigenvalues))
        explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
        components = eigenvectors[:, order[:2]]
        projection = centered @ components
    if projection.shape[1] < 2:
        projection = np.pad(projection, ((0, 0), (0, 2 - projection.shape[1])))
    if projection.shape[0] == 0:
        return None
    return eigenvalues, projection, explained


def _mean_pairwise_cosine(matrix: np.ndarray) -> Optional[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    rows = int(matrix.shape[0])
    if rows < 2:
        return None
    # Sum identity: sum_{i != j} u_i . u_j = ||sum_i u_i||^2 - sum_i ||u_i||^2.
    # This matches the pairwise definition (zero-norm rows contribute nothing)
    # without materializing the O(M^2) Gram matrix of the reduction sample;
    # normalized rows are also accumulated in bounded blocks.
    total = np.zeros(int(matrix.shape[1]), dtype=np.float64)
    squared = 0.0
    for start in range(0, rows, _MEAN_COSINE_BLOCK_ROWS):
        chunk = matrix[start : start + _MEAN_COSINE_BLOCK_ROWS]
        norms = np.linalg.norm(chunk, axis=1)
        safe = np.where(norms > 0, norms, 1.0)
        normalized = chunk / safe[:, None]
        total += normalized.sum(axis=0)
        squared += float(np.einsum("ij,ij->", normalized, normalized))
    mean = (float(total @ total) - squared) / float(rows * (rows - 1))
    if not math.isfinite(mean):
        return None
    return float(np.clip(mean, -1.0, 1.0))


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
