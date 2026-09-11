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

"""Focused coverage of the existing-index C++ AutoTune Python adapter."""

import json

import numpy as np
import pytest
import pyvsag


@pytest.fixture(scope="module")
def workload():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(128, 16)).astype(np.float32)
    queries = rng.normal(size=(8, 16)).astype(np.float32)
    ids = np.arange(128, dtype=np.int64) + 1000
    distances = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
    ground_truth = np.ascontiguousarray(ids[np.argsort(distances, axis=1)[:, :5]])
    params = json.dumps({
        "dtype": "float32", "metric_type": "l2", "dim": 16,
        "index_param": {"max_degree": 16, "ef_construction": 100,
                        "base_quantization_type": "fp32"},
    })
    index = pyvsag.Index("hgraph", params)
    index.build(base, ids, len(ids), 16)
    return index, queries, ground_truth, params


def tune(workload, **overrides):
    index, queries, ground_truth, _ = workload
    request = dict(queries=queries, ground_truth=ground_truth, top_k=5,
                   parameter_space={"hgraph": {"ef_search": [16, 128]}},
                   constraints={"recall_at_k": 0.9}, objective="latency_avg_ms")
    request.update(overrides)
    return index.autotune_search(**request)


def test_recommendation_runs_search(workload):
    index, queries, ground_truth, _ = workload
    before = index.knn_search(queries[0], 5, '{"hgraph":{"ef_search":128}}')
    result = tune(workload, include_raw_evaluation=True, concurrency=2)
    assert result["status"] == "success"
    assert json.loads(result["search_parameters"])["hgraph"]["ef_search"] in (16, 128)
    assert result["metrics"]["recall_at_k"] >= 0.9
    assert result["metrics"]["latency_avg_ms"] >= 0
    assert result["best_effort"] is None
    assert len(result["report"]["trials"]) == 2
    assert all("raw_eval_result" in trial for trial in result["report"]["trials"])
    assert result["report"]["recommendation"]["metrics"] == result["metrics"]
    ids, distances = index.knn_search(queries[0], 5, result["search_parameters"])
    assert ids.shape == distances.shape == (5,)
    assert np.isin(ids, ground_truth[0]).mean() >= 0.8
    after = index.knn_search(queries[0], 5, '{"hgraph":{"ef_search":128}}')
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_no_feasible_candidate(workload):
    result = tune(workload, constraints={"index_memory_mb": 0})
    assert result["status"] == "no_feasible_candidate"
    assert result["search_parameters"] is None
    assert result["metrics"] == {}
    assert result["best_effort"]["metrics"]["index_memory_mb"] > 0
    assert result["report"]["status"] == result["status"]


@pytest.mark.parametrize("field,dtype", [("queries", np.float64),
                                         ("ground_truth", np.int32)])
def test_wrong_dtype(workload, field, dtype):
    array = workload[1 if field == "queries" else 2]
    with pytest.raises(ValueError, match="dtype"):
        tune(workload, **{field: array.astype(dtype)})


@pytest.mark.parametrize("field", ["queries", "ground_truth"])
@pytest.mark.parametrize("layout", ["flat", "empty", "strided", "unaligned"])
def test_invalid_matrix(workload, field, layout):
    array = workload[1 if field == "queries" else 2]
    if layout == "flat":
        array = array.ravel()
    elif layout == "empty":
        array = array[:0]
    elif layout == "strided":
        array = array[:, ::-1]
    else:
        array = np.ndarray(array.shape, dtype=array.dtype,
                           buffer=bytearray(array.nbytes + 1), offset=1)
    with pytest.raises(ValueError):
        tune(workload, **{field: array})


def test_mismatched_rows(workload):
    with pytest.raises(ValueError, match="one row"):
        tune(workload, ground_truth=workload[2][:-1])


@pytest.mark.parametrize("overrides", [
    {"top_k": 0}, {"top_k": 6}, {"top_k": -1},
    {"constraints": {}}, {"constraints": {"recall_at_k": 1.1}},
    {"constraints": {"recall_at_k": float("nan")}},
    {"constraints": {"recall_at_k": True}},
    {"constraints": {"unknown": 1}}, {"objective": "unknown"},
    {"objective": "build_seconds"}, {"constraints": {"index_size_mb": 1}},
    {"concurrency": 0}, {"concurrency": 201}, {"max_trials": 0},
    {"max_trials": 1}, {"parameter_space": {"hgraph": {"ef_search": []}}},
    {"parameter_space": "{}"}, {"queries": [[0.0] * 16]},
])
def test_invalid_request(workload, overrides):
    with pytest.raises((ValueError, TypeError)):
        tune(workload, **overrides)


def test_empty_index(workload):
    _, queries, ground_truth, params = workload
    with pytest.raises(ValueError, match="empty"):
        tune((pyvsag.Index("hgraph", params), queries, ground_truth, params))


def test_execution_failure(workload):
    # Shape is valid, but the index rejects every trial due to its configured dimension.
    with pytest.raises(RuntimeError, match="autotune_search failed"):
        tune(workload, queries=np.zeros((8, 15), dtype=np.float32))


def test_non_float32_index(workload):
    _, queries, ground_truth, params = workload
    params = json.loads(params)
    params["dtype"] = "float16"
    params["index_param"]["base_quantization_type"] = "fp16"
    index = pyvsag.Index("hgraph", json.dumps(params))
    with pytest.raises(ValueError, match="float32 indexes"):
        tune((index, queries, ground_truth, params))


def test_loaded_index(workload, tmp_path):
    index, queries, ground_truth, params = workload
    path = str(tmp_path / "index.vsag")
    index.save(path)
    restored = pyvsag.Index("hgraph", params)
    restored.load(path)
    result = tune((restored, queries, ground_truth, params),
                  parameter_space={"hgraph": {"ef_search": 128}})
    assert result["status"] == "success"
    ids, _ = restored.knn_search(queries[0], 5, result["search_parameters"])
    np.testing.assert_array_equal(ids, ground_truth[0])


def test_ivf_search_tuning():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(64, 8)).astype(np.float32)
    ids = np.arange(len(base), dtype=np.int64)
    queries = base[:4].copy()
    truth = np.ascontiguousarray(np.argsort(
        ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2), axis=1
    )[:, :3], dtype=np.int64)
    index = pyvsag.Index("ivf", json.dumps({
        "dtype": "float32", "metric_type": "l2", "dim": 8,
        "index_param": {"buckets_count": 4, "partition_strategy_type": "ivf",
                        "ivf_train_type": "kmeans", "base_quantization_type": "fp32"},
    }))
    index.build(base, ids, len(base), 8)
    result = index.autotune_search(
        queries, truth, 3, {"ivf": {"scan_buckets_count": [2, 4]}},
        {"recall_at_k": 1.0}, "latency_avg_ms",
    )
    assert result["status"] == "success"
    neighbors, _ = index.knn_search(queries[0], 3, result["search_parameters"])
    np.testing.assert_array_equal(neighbors, truth[0])
