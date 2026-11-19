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

"""Tests for pyvsag.eval, the Python front end of the eval_performance measurement core."""

import json

import numpy as np
import pytest

import pyvsag
from pyvsag.eval import (
    EvalDataset,
    compute_ground_truth,
    evaluate_build,
    evaluate_search,
    export_results,
    format_results,
    num_threads_building,
    run_job,
)
from pyvsag.eval.__main__ import main

DIM = 16
TOP_K = 5
QUICK = {"search_query_count": 40}

BRUTE_FORCE_PARAMS = json.dumps(
    {
        "dim": DIM,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {},
    }
)
HGRAPH_PARAMS = json.dumps(
    {
        "dim": DIM,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100,
        },
    }
)
HGRAPH_SEARCH = {"hgraph": {"ef_search": 100}}


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(7)
    base = rng.random((300, DIM), dtype=np.float32)
    queries = rng.random((20, DIM), dtype=np.float32)
    return EvalDataset.from_arrays(base, queries, metric="l2").with_ground_truth(TOP_K)


def test_compute_ground_truth_matches_numpy():
    rng = np.random.default_rng(1)
    base = rng.random((100, 8), dtype=np.float32)
    queries = rng.random((5, 8), dtype=np.float32)
    ids, dists = compute_ground_truth(base, queries, 3, "l2", ids=np.arange(100) + 1000)
    expected = np.argsort(((queries[:, None, :] - base[None]) ** 2).sum(-1), axis=1)[:, :3]
    np.testing.assert_array_equal(ids, expected + 1000)
    assert np.all(np.diff(dists, axis=1) >= 0)


def test_build_and_search_with_recall(dataset):
    index = pyvsag.Index("hgraph", HGRAPH_PARAMS)
    build = evaluate_build(index, dataset)
    assert build["action"] == "build"
    assert build["dataset_info"]["base_count"] == 300
    assert build["tps"] > 0
    assert build["index"] == "hgraph"
    assert build["index_info"]["metric_type"] == "l2"

    search = evaluate_search(index, dataset, top_k=TOP_K, search_param=HGRAPH_SEARCH, **QUICK)
    assert search["action"] == "search"
    assert search["qps"] > 0
    assert search["recall_avg"] >= 0.95
    assert search["latency_detail(ms)"]["p99"] >= search["latency_detail(ms)"]["p50"]
    assert search["search_mode"] == "knn"


def test_brute_force_has_perfect_recall(dataset):
    index = pyvsag.Index("brute_force", BRUTE_FORCE_PARAMS)
    evaluate_build(index, dataset)
    search = evaluate_search(index, dataset, top_k=TOP_K, search_param="{}", **QUICK)
    assert search["recall_avg"] == pytest.approx(1.0)


def test_metrics_can_be_disabled(dataset):
    index = pyvsag.Index("brute_force", BRUTE_FORCE_PARAMS)
    evaluate_build(index, dataset)
    search = evaluate_search(
        index,
        dataset,
        top_k=TOP_K,
        search_param="{}",
        enable_recall=False,
        enable_percent_recall=False,
        enable_percent_latency=False,
        **QUICK,
    )
    assert "recall_avg" not in search
    assert "latency_detail(ms)" not in search
    assert search["qps"] > 0


def test_search_without_ground_truth_requires_disabling_recall(dataset):
    plain = EvalDataset.from_arrays(dataset.base, dataset.queries)
    index = pyvsag.Index("brute_force", BRUTE_FORCE_PARAMS)
    evaluate_build(index, plain)
    with pytest.raises(ValueError):
        evaluate_search(index, plain, top_k=TOP_K, search_param="{}", **QUICK)
    search = evaluate_search(
        index,
        plain,
        top_k=TOP_K,
        search_param="{}",
        enable_recall=False,
        enable_percent_recall=False,
        **QUICK,
    )
    assert search["qps"] > 0


def test_invalid_options_are_reported(dataset):
    index = pyvsag.Index("brute_force", BRUTE_FORCE_PARAMS)
    with pytest.raises(ValueError, match="unknown"):
        evaluate_build(index, dataset, no_such_option=1)
    evaluate_build(index, dataset)
    with pytest.raises(ValueError):
        evaluate_search(index, dataset, top_k=TOP_K + 1, search_param="{}", **QUICK)
    with pytest.raises(ValueError):
        evaluate_search(index, dataset, top_k=0, search_param="{}", **QUICK)
    with pytest.raises(TypeError):
        evaluate_search(object(), dataset, top_k=TOP_K)


def test_dataset_validation():
    base = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        EvalDataset.from_arrays(base, np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        EvalDataset.from_arrays(base, base, metric="hamming")
    with pytest.raises(ValueError):
        EvalDataset.from_arrays(base, base, np.zeros((3, 2), dtype=np.int64))


def test_num_threads_building_restores_previous_value():
    from pyvsag.eval import _native

    original = _native.set_num_threads_building(3)
    try:
        with num_threads_building(5):
            assert _native.set_num_threads_building(5) == 5
        assert _native.set_num_threads_building(3) == 3
    finally:
        _native.set_num_threads_building(original)


def test_formatters():
    result = {
        "case": {
            "index": "hgraph",
            "dataset_info": {"base_count": 10, "dim": 4, "data_type": "float32"},
            "duration(s)": 1.5,
            "tps": 6.0,
            "qps": 100.0,
            "latency_avg(ms)": 0.5,
            "latency_detail(ms)": {"p90": 1.0, "p95": 1.5, "p99": 2.0},
            "recall_avg": 0.9,
        }
    }
    assert json.loads(format_results(result, "json")) == result
    table = format_results(result, "table")
    assert "RecallAvg" in table and "hgraph" in table and "0.900000" in table
    assert format_results(result, "text") == table
    line = format_results(result, "line_protocol")
    assert line.startswith("performance,case_name=case,index_name=hgraph,dtype=float32 ")
    assert "qps=100.000000" in line and line.endswith("\n")
    with pytest.raises(ValueError):
        format_results(result, "xml")


def test_export_to_files(tmp_path):
    result = {"case": {"index": "hgraph"}}
    target = tmp_path / "out.json"
    assert export_results(result, "json", "file://" + str(target))
    assert json.loads(target.read_text()) == result
    appended = tmp_path / "append.txt"
    for _ in range(2):
        export_results(result, "line_protocol", "appendfile://" + str(appended))
    assert len(appended.read_text().splitlines()) == 2
    with pytest.raises(ValueError):
        export_results(result, "json", "ftp://host/file")


@pytest.fixture
def hdf5_path(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(3)
    base = rng.random((300, DIM), dtype=np.float32)
    queries = rng.random((20, DIM), dtype=np.float32)
    neighbors, distances = compute_ground_truth(base, queries, 10, "l2")
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("train", data=base)
        file.create_dataset("test", data=queries)
        file.create_dataset("neighbors", data=neighbors)
        file.create_dataset("distances", data=distances)
        file.attrs["distance"] = "euclidean"
    return str(path)


def test_from_hdf5(hdf5_path):
    loaded = EvalDataset.from_hdf5(hdf5_path)
    assert (loaded.num_base, loaded.num_queries, loaded.dim) == (300, 20, DIM)
    assert loaded.metric == "l2"
    assert loaded.ground_truth.shape == (20, 10)


def _case(hdf5_path, action, **extra):
    case = {
        "datapath": hdf5_path,
        "type": action,
        "index_name": "hgraph",
        "create_params": HGRAPH_PARAMS,
        "search_params": json.dumps(HGRAPH_SEARCH),
        "topk": TOP_K,
        "search_query_count": 40,
    }
    case.update(extra)
    return case


def test_run_job_build_search(hdf5_path):
    job = {"global": {"num_threads_building": 2}, "c": _case(hdf5_path, "build,search")}
    results = run_job(job)
    result = results["c"]
    assert result["action"] == "build,search"
    assert result["tps"] > 0 and result["qps"] > 0
    assert result["recall_avg"] >= 0.95


def test_run_job_build_then_search_from_saved_index(hdf5_path, tmp_path):
    index_path = str(tmp_path / "nested" / "index")
    results = run_job(
        {
            "build": _case(hdf5_path, "build", index_path=index_path),
            "search": _case(
                hdf5_path, "search", index_path=index_path, delete_index_after_search=True
            ),
        }
    )
    assert results["build"]["action"] == "build"
    assert results["search"]["recall_avg"] >= 0.95
    assert not (tmp_path / "nested" / "index").exists()


def test_run_job_rejects_bad_cases(hdf5_path):
    with pytest.raises(ValueError, match="missing required key"):
        run_job({"c": {"type": "build"}})
    with pytest.raises(ValueError, match="search_params"):
        case = _case(hdf5_path, "search")
        del case["search_params"]
        run_job({"c": case})
    with pytest.raises(ValueError, match="type must be"):
        run_job({"c": _case(hdf5_path, "train")})


def test_cli_with_yaml_job(hdf5_path, tmp_path, capsys):
    yaml = pytest.importorskip("yaml")
    output = tmp_path / "result.json"
    job = {
        "global": {
            "exporters": {
                "to-file": {"format": "json", "to": "file://" + str(output)},
                "to-stdout": {"format": "table", "to": "stdout"},
            }
        },
        "case1": _case(hdf5_path, "build,search"),
    }
    job_path = tmp_path / "job.yaml"
    job_path.write_text(yaml.safe_dump(job))

    assert main([str(job_path)]) == 0
    assert "case1" in capsys.readouterr().out
    assert json.loads(output.read_text())["case1"]["recall_avg"] >= 0.95
    assert main([str(tmp_path / "missing.yaml")]) == 1
