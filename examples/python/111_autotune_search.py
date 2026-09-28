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

import json

import numpy as np
import pyvsag


rng = np.random.default_rng(42)
base = rng.normal(size=(256, 32)).astype(np.float32)
queries = rng.normal(size=(16, 32)).astype(np.float32)
ids = np.arange(len(base), dtype=np.int64)
top_k = 10
squared_distances = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
ground_truth = np.ascontiguousarray(
    ids[np.argsort(squared_distances, axis=1)[:, :top_k]]
)

index = pyvsag.Index("hgraph", json.dumps({
    "dtype": "float32", "metric_type": "l2", "dim": base.shape[1],
    "index_param": {"max_degree": 16, "ef_construction": 100},
}))
index.build(base, ids, len(base), base.shape[1])
result = index.autotune_search(
    queries=queries,
    ground_truth=ground_truth,
    top_k=top_k,
    parameter_space={"hgraph": {"ef_search": [16, 32, 128]}},
    constraints={"recall_at_k": 0.95},
    objective="latency_avg_ms",
)
print("Status:", result["status"])
if result["status"] == "success":
    print("Search parameters:", result["search_parameters"])
    print("Validated metrics:", result["metrics"])
    neighbors, distances = index.knn_search(
        queries[0], top_k, result["search_parameters"]
    )
    print("Neighbor IDs:", neighbors)
    print("Distances:", distances)
else:
    print("Best effort (constraints not met):", result["best_effort"])
