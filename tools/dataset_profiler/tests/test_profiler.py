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

"""Tests for the deterministic profiling pipeline."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from dataset_profiler.profiler import (
    _DENSE_BLOCK_BYTES,
    _DENSE_BLOCK_BYTES_PER_VALUE,
    _DENSE_BLOCK_MAX_ROWS,
    _DimAccumulator,
    _dense_block_rows,
    _dense_euclidean,
    _mean_pairwise_cosine,
    _outliers_metric,
    profile_dataset,
)
from dataset_profiler.report import DEFINITIONS
from dataset_profiler.sampling import make_rng, sample_indices
from dataset_profiler.stats import dump_json

EXPECTED_DENSE_METRICS = set(DEFINITIONS)
REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "tool",
    "input",
    "dataset",
    "config",
    "metrics",
    "warnings",
}


def _write_dense(path, matrix, metric="euclidean"):
    import h5py

    with h5py.File(str(path), "w") as handle:
        handle.attrs["distance"] = metric
        handle.create_dataset("train", data=np.asarray(matrix, dtype=np.float32))


def _write_sparse(path, records, metric="ip"):
    import h5py

    from conftest import encode_sparse_record

    with h5py.File(str(path), "w") as handle:
        handle.attrs["type"] = "sparse"
        handle.attrs["distance"] = metric
        stream = b"".join(
            encode_sparse_record(ids, values) for ids, values in records
        )
        handle.create_dataset(
            "train", data=np.frombuffer(stream, dtype=np.int8).copy()
        )


def test_dense_report_contains_all_documented_metrics(dense_path):
    report = profile_dataset(str(dense_path), sample_size=32, pairwise_sample_size=16)
    assert set(report) == REQUIRED_TOP_LEVEL_KEYS
    assert report["schema_version"] == "1.0"
    assert report["tool"]["name"] == "vsag-dataset-profiler"
    assert set(report["metrics"]) == EXPECTED_DENSE_METRICS
    json.loads(dump_json(report))  # must be JSON serializable
    for name, metric in report["metrics"].items():
        assert metric["definition"] == DEFINITIONS[name]
        assert metric["scope"] in ("full", "sample", "mixed")


def test_dense_full_metrics_match_source_fixture(dense_path, dense_matrix):
    report = profile_dataset(str(dense_path), sample_size=200, pairwise_sample_size=200)
    metrics = report["metrics"]
    valid = np.where(np.isfinite(dense_matrix), dense_matrix, 0.0).astype(np.float64)
    norms = np.linalg.norm(valid, axis=1)
    assert metrics["norms"]["stats"]["count"] == dense_matrix.shape[0]
    assert metrics["norms"]["non_finite_norms"] == 0
    assert math.isclose(metrics["norms"]["stats"]["mean"], float(norms.mean()), rel_tol=1e-9)

    zero_values = int(np.count_nonzero((dense_matrix == 0) & np.isfinite(dense_matrix)))
    assert metrics["sparsity"]["total_values"] == dense_matrix.size
    assert metrics["sparsity"]["zero_values"] == zero_values
    assert metrics["invalid_values"]["invalid_values"] == 2
    assert metrics["invalid_values"]["vectors_with_invalid"] == 1

    # rows 0/2 and 1/5 are exact duplicates plus the duplicated sparse-style pair
    exact = metrics["duplicates"]["exact"]
    assert exact["duplicate_vectors"] == 2
    assert exact["unique_vectors"] == dense_matrix.shape[0] - 2


def test_pairwise_matrix_is_bounded_by_sample_size(dense_path):
    report = profile_dataset(str(dense_path), sample_size=20, pairwise_sample_size=8)
    pairwise = report["metrics"]["pairwise_distances"]
    assert pairwise["pair_count"] == 8 * 7 // 2
    assert pairwise["sample_size"] == 8
    assert pairwise["matrix_is_bounded"] is True
    assert report["config"]["effective_sample_size"] == 20
    assert report["config"]["effective_pairwise_sample_size"] == 8


def test_two_runs_with_same_seed_are_identical(dense_path):
    first = profile_dataset(str(dense_path), seed=7, sample_size=24, pairwise_sample_size=12)
    second = profile_dataset(str(dense_path), seed=7, sample_size=24, pairwise_sample_size=12)
    assert dump_json(first) == dump_json(second)


def test_different_seed_changes_sampling_metadata(dense_path):
    kwargs = dict(sample_size=20, pairwise_sample_size=20)
    first = profile_dataset(str(dense_path), seed=1, **kwargs)
    second = profile_dataset(str(dense_path), seed=2, **kwargs)
    assert first["metrics"]["pairwise_distances"]["stats"] != second["metrics"]["pairwise_distances"]["stats"]


def test_metric_alias_and_file_distance_attribute(dense_path):
    # the fixture stores distance="l2" which must resolve to euclidean
    report = profile_dataset(str(dense_path), sample_size=8, pairwise_sample_size=8)
    assert report["config"]["metric"] == "euclidean"
    report = profile_dataset(str(dense_path), metric="cosine", sample_size=8, pairwise_sample_size=8)
    assert report["config"]["metric"] == "angular"


def test_unsupported_metric_rejected(dense_path):
    with pytest.raises(ValueError, match="unsupported distance metric"):
        profile_dataset(str(dense_path), metric="hamming")


def test_invalid_config_rejected(dense_path):
    with pytest.raises(ValueError):
        profile_dataset(str(dense_path), sample_size=0)
    with pytest.raises(ValueError):
        profile_dataset(str(dense_path), pairwise_sample_size=-3)
    with pytest.raises(ValueError):
        profile_dataset(str(dense_path), clusters=1001)


def test_anisotropy_and_clustering_on_dense(dense_path):
    report = profile_dataset(
        str(dense_path), sample_size=32, pairwise_sample_size=16, clusters=4, seed=5
    )
    aniso = report["metrics"]["anisotropy"]
    assert aniso["available"] is True
    assert 0.0 <= aniso["top1_eigenvalue_share"] <= 1.0
    assert aniso["effective_rank"] > 1.0
    clustering = report["metrics"]["clustering"]
    assert clustering["available"] is True
    assert clustering["clustering_available"] is True
    assert sum(clustering["cluster_sizes"]) == 32
    assert len(clustering["labels"]) == 32
    assert len(clustering["projection_points"]) == 32


def test_clustering_disabled_by_default(dense_path):
    report = profile_dataset(str(dense_path), sample_size=16, pairwise_sample_size=8)
    clustering = report["metrics"]["clustering"]
    assert clustering["available"] is True
    assert clustering["clustering_available"] is False


def test_outliers_flag_huge_norm_row(dense_path):
    report = profile_dataset(str(dense_path), sample_size=8, pairwise_sample_size=8)
    outliers = report["metrics"]["outliers"]
    assert outliers["outlier_count"] >= 1
    assert outliers["method"].startswith("modified z-score")


def test_sparse_report_metrics_and_determinism(sparse_path, sparse_records):
    first = profile_dataset(str(sparse_path), sample_size=16, pairwise_sample_size=8, seed=3)
    second = profile_dataset(str(sparse_path), sample_size=16, pairwise_sample_size=8, seed=3)
    assert dump_json(first) == dump_json(second)

    metrics = first["metrics"]
    assert set(metrics) == EXPECTED_DENSE_METRICS
    assert first["dataset"]["vector_type"] == "sparse"
    assert first["dataset"]["count"] == len(sparse_records)
    assert first["dataset"]["dim_inferred"] is True
    assert metrics["sparsity"]["representation"] == "sparse"
    assert metrics["sparsity"]["density"] is not None
    assert metrics["duplicates"]["exact"]["duplicate_vectors"] >= 1
    assert metrics["pairwise_distances"]["pair_count"] == 28


def test_sparse_without_offsets_matches_offsets_stream(
    sparse_path, sparse_no_offsets_path
):
    kwargs = dict(seed=11, sample_size=16, pairwise_sample_size=8)
    with_offsets = profile_dataset(str(sparse_path), **kwargs)
    without = profile_dataset(str(sparse_no_offsets_path), **kwargs)
    # full-dataset statistics must agree between the two encodings
    for key in ("norms", "sparsity", "dimensions", "duplicates", "invalid_values"):
        assert with_offsets["metrics"][key] == without["metrics"][key], key
    assert any("no '/train_offsets'" in w for w in without["warnings"])


def test_sparse_non_ip_metric_warns(sparse_path):
    report = profile_dataset(
        str(sparse_path), metric="euclidean", sample_size=8, pairwise_sample_size=8
    )
    assert report["config"]["metric"] == "euclidean"
    assert any("sparse VSAG evaluation supports only" in w for w in report["warnings"])


def test_no_distance_attribute_falls_back_with_warning(tmp_path, dense_matrix):
    import h5py

    path = tmp_path / "no_distance.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("train", data=dense_matrix)
    report = profile_dataset(str(path), sample_size=8, pairwise_sample_size=8)
    assert report["config"]["metric"] == "euclidean"
    assert any("no distance metric was provided" in w for w in report["warnings"])


def test_empty_sparse_dataset_rejected(tmp_path):
    import h5py

    path = tmp_path / "empty_sparse.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["type"] = "sparse"
        f.create_dataset("train", data=np.zeros(0, dtype=np.int8))
    with pytest.raises(Exception, match="empty"):
        profile_dataset(str(path))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        profile_dataset(str(tmp_path / "nope.hdf5"))


def test_split_selection(dense_path):
    report = profile_dataset(str(dense_path), split="test", sample_size=4, pairwise_sample_size=4)
    assert report["dataset"]["count"] == 6


def test_nearest_neighbor_uses_distances_not_coordinates(tmp_path):
    path = tmp_path / "nearest.hdf5"
    _write_dense(path, [[0, 0], [3, 4], [6, 8]])
    report = profile_dataset(
        str(path), metric="euclidean", sample_size=3, pairwise_sample_size=3
    )
    stats = report["metrics"]["nearest_neighbor_distances"]["stats"]
    assert stats["count"] == 3
    assert stats["min"] == 5.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 5.0


@pytest.mark.parametrize("vector_type", ["dense", "sparse"])
def test_pairwise_subset_is_seeded_random_subset(tmp_path, vector_type):
    seed = 17
    rng = make_rng(seed)
    expected_sample = sample_indices(10, 6, rng)
    expected_positions = np.sort(rng.permutation(6)[:2])
    assert expected_positions.tolist() != [0, 1]
    expected_rows = expected_sample[expected_positions]
    expected_distance = float(abs(2 ** expected_rows[0] - 2 ** expected_rows[1]))

    path = tmp_path / (vector_type + ".hdf5")
    values = (2.0 ** np.arange(10)).astype(np.float32)
    if vector_type == "dense":
        _write_dense(path, values.reshape(-1, 1))
    else:
        records = [
            (np.array([0], dtype=np.uint32), np.array([value], dtype=np.float32))
            for value in values
        ]
        _write_sparse(path, records, metric="euclidean")
    report = profile_dataset(
        str(path),
        metric="euclidean",
        seed=seed,
        sample_size=6,
        pairwise_sample_size=2,
    )
    stats = report["metrics"]["pairwise_distances"]["stats"]
    assert stats["count"] == 1
    assert stats["min"] == expected_distance


def test_sparse_dimensions_include_implicit_zeros(tmp_path):
    path = tmp_path / "implicit_zeros.hdf5"
    records = [
        (np.array([0], dtype=np.uint32), np.array([2], dtype=np.float32)),
        (np.array([1], dtype=np.uint32), np.array([4], dtype=np.float32)),
    ]
    _write_sparse(path, records)
    report = profile_dataset(str(path), sample_size=2, pairwise_sample_size=2)
    per_dimension = report["metrics"]["dimensions"]["per_dimension"]
    assert per_dimension["mean"]["values"] == [1.0, 2.0]
    assert per_dimension["std"]["values"] == [1.0, 2.0]
    assert per_dimension["min"]["values"] == [0.0, 0.0]
    assert per_dimension["max"]["values"] == [2.0, 4.0]
    assert per_dimension["zero_ratio"]["values"] == [0.5, 0.5]


def test_sparse_huge_feature_id_skips_only_dimension_arrays(tmp_path):
    path = tmp_path / "huge_feature_id.hdf5"
    records = [
        (np.array([0], dtype=np.uint32), np.array([1], dtype=np.float32)),
        (np.array([2**31], dtype=np.uint32), np.array([2], dtype=np.float32)),
    ]
    _write_sparse(path, records)
    report = profile_dataset(str(path), sample_size=2, pairwise_sample_size=2)
    assert report["dataset"]["dim"] == 2**31 + 1
    assert report["metrics"]["dimensions"]["available"] is False
    assert "exceeds the per-dimension statistics limit" in report["metrics"]["dimensions"]["reason"]
    assert report["metrics"]["norms"]["count"] == 2
    assert report["metrics"]["sparsity"]["cells"] == 2 * (2**31 + 1)
    assert any("other metrics are still reported" in item for item in report["warnings"])


def test_zero_mad_and_iqr_still_detect_extreme_norms():
    norms = np.ones(100, dtype=np.float64)
    norms[-1] = 100.0
    result = _outliers_metric(norms)
    assert result["outlier_count"] == 1
    assert result["method"] == "median fallback (MAD and IQR are zero)"
    identical = _outliers_metric(np.ones(100, dtype=np.float64))
    assert identical["outlier_count"] == 0
    assert identical["method"].startswith("all norms identical")


def test_ip_near_duplicates_use_nonnegative_duplicate_distance(tmp_path):
    path = tmp_path / "ip_duplicates.hdf5"
    _write_dense(path, [[2, 0], [2, 0], [100, 0]], metric="ip")
    report = profile_dataset(
        str(path), metric="ip", sample_size=3, pairwise_sample_size=3
    )
    near = report["metrics"]["duplicates"]["near_duplicate"]
    assert near["metric"] == "euclidean"
    assert near["pair_count"] == 3
    assert near["near_duplicate_pairs"] == 1


def test_dense_euclidean_avoids_catastrophic_cancellation(tmp_path):
    path = tmp_path / "large_offset.hdf5"
    _write_dense(path, [[1e10, 0], [1e10, 1], [1e10, 2]])
    report = profile_dataset(
        str(path), metric="euclidean", sample_size=3, pairwise_sample_size=3
    )
    pairwise = report["metrics"]["pairwise_distances"]["stats"]
    assert pairwise["min"] == 1.0
    assert pairwise["max"] == 2.0
    assert pairwise["mean"] == pytest.approx(4.0 / 3.0)
    nearest = report["metrics"]["nearest_neighbor_distances"]["stats"]
    assert nearest["min"] == 1.0
    assert nearest["max"] == 1.0
    near = report["metrics"]["duplicates"]["near_duplicate"]
    assert near["near_duplicate_pairs"] == 0


def test_sparse_euclidean_avoids_catastrophic_cancellation(tmp_path):
    path = tmp_path / "sparse_large_offset.hdf5"
    base = 2 ** 33
    records = [
        (np.array([0], dtype=np.uint32), np.array([base], dtype=np.float32)),
        (np.array([0], dtype=np.uint32), np.array([base + 2048], dtype=np.float32)),
        (np.array([0], dtype=np.uint32), np.array([base + 4096], dtype=np.float32)),
    ]
    _write_sparse(path, records, metric="euclidean")
    report = profile_dataset(
        str(path), metric="euclidean", sample_size=3, pairwise_sample_size=3
    )
    pairwise = report["metrics"]["pairwise_distances"]["stats"]
    assert pairwise["min"] == 2048.0
    assert pairwise["max"] == 4096.0
    assert pairwise["mean"] == pytest.approx((2048.0 + 2048.0 + 4096.0) / 3.0)


def test_dense_block_rows_are_bounded_by_working_bytes():
    assert _dense_block_rows(0) == _DENSE_BLOCK_MAX_ROWS
    assert _dense_block_rows(8) == _DENSE_BLOCK_MAX_ROWS
    wide = _dense_block_rows(4096)
    assert wide < _DENSE_BLOCK_MAX_ROWS
    # raw + float64 copy + boolean flags + float64 temporaries stay in budget
    assert wide * 4096 * _DENSE_BLOCK_BYTES_PER_VALUE <= _DENSE_BLOCK_BYTES
    assert _dense_block_rows(10 ** 9) == 1


def test_mean_pairwise_cosine_matches_explicit_pairs():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    expected = (1.0 + -1.0 + 0.0 + -1.0 + 0.0 + 0.0) / 6.0
    assert _mean_pairwise_cosine(matrix) == pytest.approx(expected)
    assert _mean_pairwise_cosine(np.ones((1, 2))) is None


def test_sparse_implicit_zeros_still_exclude_invalid_entries(tmp_path):
    path = tmp_path / "implicit_zeros_invalid.hdf5"
    records = [
        (np.array([0], dtype=np.uint32), np.array([2], dtype=np.float32)),
        (np.array([0], dtype=np.uint32), np.array([np.nan], dtype=np.float32)),
        (np.array([1], dtype=np.uint32), np.array([4], dtype=np.float32)),
    ]
    _write_sparse(path, records)
    report = profile_dataset(str(path), sample_size=3, pairwise_sample_size=3)
    per_dimension = report["metrics"]["dimensions"]["per_dimension"]
    assert per_dimension["mean"]["values"][0] == 1.0
    assert per_dimension["std"]["values"][0] == 1.0
    assert per_dimension["min"]["values"][0] == 0.0
    assert per_dimension["max"]["values"][0] == 2.0
    assert per_dimension["zero_ratio"]["values"][0] == 0.5
    assert report["metrics"]["invalid_values"]["invalid_values"] == 1


def test_sparse_accumulator_rejects_id_driven_oversized_allocation():
    accumulator = _DimAccumulator()
    with pytest.raises(ValueError, match="exceeds per-dimension statistics limit"):
        accumulator._ensure(2**31 + 1)
    assert accumulator.count.size == 0


def test_zero_robust_spread_detects_deviations_on_both_sides():
    norms = np.ones(101, dtype=np.float64)
    norms[0] = 0.0
    norms[-1] = 1000.0
    result = _outliers_metric(norms)
    assert result["outlier_count"] == 2
    assert result["method"] == "median fallback (MAD and IQR are zero)"


def test_sparse_ip_near_duplicates_use_euclidean_distance(tmp_path):
    path = tmp_path / "sparse_ip_duplicates.hdf5"
    records = [
        (np.array([0], dtype=np.uint32), np.array([0.5], dtype=np.float32)),
        (np.array([0], dtype=np.uint32), np.array([0.5], dtype=np.float32)),
        (np.array([0], dtype=np.uint32), np.array([2], dtype=np.float32)),
        (
            np.array([0, 1], dtype=np.uint32),
            np.array([2, 100], dtype=np.float32),
        ),
    ]
    _write_sparse(path, records, metric="ip")
    report = profile_dataset(str(path), sample_size=4, pairwise_sample_size=4)
    near = report["metrics"]["duplicates"]["near_duplicate"]
    assert near["metric"] == "euclidean"
    assert near["median_pairwise_distance"] > 0
    assert near["near_duplicate_pairs"] == 1


def test_large_constant_norm_histogram_has_representable_edges(tmp_path):
    path = tmp_path / "constant_large_norm.hdf5"
    value = float(2**60)
    _write_dense(path, [[value, 0]])
    report = profile_dataset(str(path), sample_size=1, pairwise_sample_size=1)
    histogram = report["metrics"]["norms"]["histogram"]
    assert sum(histogram["counts"]) == 1
    assert histogram["bins"][0] < value < histogram["bins"][-1]


@pytest.mark.parametrize("vector_type", ["dense", "sparse"])
def test_dimension_variance_is_stable_at_large_offsets(tmp_path, vector_type):
    path = tmp_path / ("stable_variance_" + vector_type + ".hdf5")
    values = np.full(256, 2**30, dtype=np.float32)
    values[-1] += 128
    if vector_type == "dense":
        _write_dense(path, values.reshape(-1, 1))
    else:
        records = [
            (np.array([0], dtype=np.uint32), np.array([value], dtype=np.float32))
            for value in values
        ]
        _write_sparse(path, records)
    report = profile_dataset(str(path), sample_size=8, pairwise_sample_size=4)
    per_dimension = report["metrics"]["dimensions"]["per_dimension"]
    assert per_dimension["mean"]["values"] == [2**30 + 0.5]
    assert per_dimension["std"]["values"][0] == pytest.approx(math.sqrt(63.75))


def test_dense_euclidean_skips_direct_fallback_for_self_distances(monkeypatch):
    real_einsum = np.einsum
    row_wise_calls = 0

    def tracking_einsum(subscripts, *operands, **kwargs):
        nonlocal row_wise_calls
        if subscripts == "ij,ij->i":
            row_wise_calls += 1
        return real_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(np, "einsum", tracking_einsum)
    distances = _dense_euclidean(np.eye(4, dtype=np.float64))
    assert row_wise_calls == 1  # only the initial squared-norm reduction
    expected = np.full((4, 4), np.sqrt(2.0))
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(distances, expected)

