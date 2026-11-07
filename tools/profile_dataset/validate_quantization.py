#!/usr/bin/env python3
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

"""Cross-check the quantization section of a dense profile_dataset report against VSAG itself.

Selects the same sampled base rows and queries as the C++ tool (see validate_profile.py), builds
a pyvsag brute_force index for every simulated quantizer and compares recall@k with the report.

Usage: python3 validate_quantization.py DATA.hdf5 REPORT.json [--tolerance 0.01]
Requires numpy, h5py and pyvsag. Exits with 1 when a recall differs beyond the tolerance.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pyvsag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_profile import Rng, derive_seed, floyd, shuffle  # noqa: E402


def analysis_rows(data, split, config, metric, begin, count):
    """Valid sampled rows [begin, begin + count) in the seeded order, as AnalysisRows()."""
    seed = config["seed"]
    sample = floyd(data.shape[0], config["sample_size"], Rng(derive_seed(seed, "sample/" + split)))
    order = list(range(len(sample)))
    shuffle(order, Rng(derive_seed(seed, "order/" + split)))
    rows = data[sample].astype(np.float32)
    finite = np.isfinite(rows).all(axis=1)
    norms = np.sqrt((rows.astype(np.float64) ** 2).sum(axis=1))
    usable = [p for p in order if finite[p] and (metric != "cosine" or norms[p] > 0)]
    picked = rows[usable[begin:begin + count]]
    if metric == "cosine":
        picked_norms = np.sqrt((picked.astype(np.float64) ** 2).sum(axis=1))
        picked = (picked.astype(np.float64) / picked_norms[:, None]).astype(np.float32)
    return picked


def exact_neighbors(base, queries, metric, k):
    """Exact top k under the VSAG distance, ties broken by the lower row id."""
    rows = base.astype(np.float64)
    result = []
    for query in queries.astype(np.float64):
        if metric == "l2":
            distance = ((rows - query) ** 2).sum(axis=1)
        else:
            distance = 1.0 - rows @ query
        order = np.lexsort((np.arange(distance.size), distance))
        result.append(set(order[:k].tolist()))
    return result


def vsag_neighbors(base, queries, metric, k, quantization):
    dim = base.shape[1]
    params = {"dtype": "float32", "metric_type": metric, "dim": dim,
              "index_param": {"base_quantization_type": quantization}}
    index = pyvsag.Index("brute_force", json.dumps(params))
    index.build(vectors=base, ids=list(range(base.shape[0])), num_elements=base.shape[0], dim=dim)
    result = []
    for query in queries:
        ids, _ = index.knn_search(vector=query, k=k, parameters=json.dumps({"bruteforce": {}}))
        result.append(set(int(i) for i in ids))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data")
    parser.add_argument("report")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    report = json.load(open(args.report))
    config = report["config"]
    metric = config["metric"]
    section = report["splits"]["train"].get("quantization")
    if section is None:
        print("report has no quantization section")
        return 1
    base_rows, query_rows, k = section["base_rows"], section["query_rows"], section["k"]
    with h5py.File(args.data, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        base = analysis_rows(train, "train", config, metric, 0, base_rows)
        if section["query_source"].startswith("sampled test rows"):
            test = np.asarray(f["test"], dtype=np.float32)
            queries = analysis_rows(test, "test", config, metric, 0, query_rows)
        else:
            queries = analysis_rows(train, "train", config, metric, base_rows, query_rows)
    truth = exact_neighbors(base, queries, metric, k)
    failures = 0
    print(f"metric {metric}, base {base.shape[0]} rows, {queries.shape[0]} queries, k {k}")
    for quantization, scheme in section["schemes"].items():
        found = vsag_neighbors(base, queries, metric, k, quantization)
        measured = float(np.mean([len(t & f) / k for t, f in zip(truth, found)]))
        simulated = scheme["recall_at_k"]
        ok = abs(simulated - measured) <= args.tolerance
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {quantization:12s} simulated {simulated:.4f}  "
              f"pyvsag brute_force {measured:.4f}")
    print(f"{len(section['schemes']) - failures}/{len(section['schemes'])} quantizers agree "
          f"(pyvsag {getattr(pyvsag, '__version__', 'unknown')})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
