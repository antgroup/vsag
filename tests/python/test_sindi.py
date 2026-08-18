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

"""Pybind statistics tests for the maintained sparse SINDI index."""

import json

import numpy as np
import pyvsag


def test_sindi_knn_search_with_statistics():
    """Expose one statistics JSON result per sparse CSR query."""
    index = pyvsag.Index(
        "sindi",
        json.dumps(
            {
                "dim": 16,
                "dtype": "sparse",
                "metric_type": "ip",
                "index_param": {
                    "use_reorder": True,
                    "doc_prune_ratio": 0.0,
                    "window_size": 10000,
                    "term_id_limit": 16,
                },
            }
        ),
    )
    index_pointers = np.array([0, 2, 4, 6], dtype=np.uint32)
    indices = np.array([0, 1, 0, 2, 1, 3], dtype=np.uint32)
    values = np.array([1.0, 0.5, 0.8, 0.7, 0.9, 0.6], dtype=np.float32)
    index.build(index_pointers, indices, values, np.arange(3, dtype=np.int64))

    query_pointers = np.array([0, 2, 4], dtype=np.uint32)
    query_indices = np.array([0, 1, 1, 3], dtype=np.uint32)
    query_values = np.array([1.0, 0.5, 0.9, 0.6], dtype=np.float32)
    search_parameters = json.dumps(
        {
            "sindi": {
                "n_candidate": 3,
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0,
            }
        }
    )

    legacy_ids, legacy_distances = index.knn_search(
        query_pointers, query_indices, query_values, 2, search_parameters
    )
    ids, distances, statistics_json = index.knn_search_with_statistics(
        query_pointers, query_indices, query_values, 2, search_parameters
    )
    assert ids.shape == legacy_ids.shape == (2, 2)
    assert distances.shape == legacy_distances.shape == (2, 2)
    assert len(statistics_json) == 2
    for value in statistics_json:
        statistics = json.loads(value)
        assert statistics["distance_evaluations"] > 0
        assert statistics["complete"] is True
