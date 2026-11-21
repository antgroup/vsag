#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Measure an index from Python with ``pyvsag.eval`` (the ``eval_performance`` core).

Covers: in-memory datasets, exact ground truth, build / search measurements, per-call options,
comparing two indexes and formatting / exporting the results.
"""

import json
import os
import tempfile

import numpy as np

import pyvsag
from pyvsag.eval import (
    EvalDataset,
    evaluate_build,
    evaluate_search,
    export_results,
    format_results,
    num_threads_building,
)

DIM = 64
TOP_K = 10


def make_dataset():
    rng = np.random.default_rng(42)
    base = rng.random((5000, DIM), dtype=np.float32)
    queries = rng.random((200, DIM), dtype=np.float32)
    # Without ground truth only QPS / latency are reported; with_ground_truth(k) computes the
    # exact neighbours with numpy so that recall is available as well.
    return EvalDataset.from_arrays(base, queries, metric="l2").with_ground_truth(TOP_K)


def index_params(index_param):
    return json.dumps(
        {"dim": DIM, "dtype": "float32", "metric_type": "l2", "index_param": index_param}
    )


def eval_hgraph(dataset):
    params = index_params(
        {"base_quantization_type": "fp32", "max_degree": 32, "ef_construction": 200}
    )
    # num_threads_building is process-wide and must be set *before* the index is created.
    with num_threads_building(4):
        index = pyvsag.Index("hgraph", params)
        build = evaluate_build(index, dataset)

    # search_param may be a dict or a JSON string; options mirror the eval_performance keys.
    search = evaluate_search(
        index,
        dataset,
        top_k=TOP_K,
        search_param={"hgraph": {"ef_search": 100}},
        num_threads_searching=1,
        search_query_count=200,
    )
    return {"hgraph": {**build, **search}}


def eval_brute_force(dataset):
    index = pyvsag.Index("brute_force", index_params({}))
    # Metrics can be switched off individually, e.g. skip memory sampling while building.
    build = evaluate_build(index, dataset, enable_memory=False)
    search = evaluate_search(index, dataset, top_k=TOP_K, search_param="{}")
    return {"brute_force": {**build, **search}}


def main():
    dataset = make_dataset()
    print(dataset)

    results = {}
    results.update(eval_hgraph(dataset))
    results.update(eval_brute_force(dataset))

    hgraph = results["hgraph"]
    print("hgraph: build %.2fs, qps %.0f, recall %.4f" % (
        hgraph["duration(s)"], hgraph["qps"], hgraph["recall_avg"]))

    # The result dicts use the same keys as the JSON printed by eval_performance.
    print(format_results(results, "table"))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "result.json")
        export_results(results, "json", "file://" + path)
        with open(path, "r", encoding="utf-8") as file:
            print("exported cases:", list(json.load(file)))


if __name__ == "__main__":
    main()
