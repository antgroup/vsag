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

from dataset_profiler.profiler import profile_dataset
from dataset_profiler.report import DEFINITIONS
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
