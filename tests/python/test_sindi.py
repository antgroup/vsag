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

"""Pybind statistics tests for the maintained sparse SINDI indexes."""

import json

import numpy as np
import pyvsag
import pytest


def assert_distance_statistics(statistics):
    total = statistics["distance_evaluations"]
    phases = statistics["distance_evaluations_by_phase"]
    backends = statistics["distance_evaluations_by_backend"]
    assert isinstance(statistics["complete"], bool)
    assert set(phases) == {"routing", "approximate", "rerank"}
    for count in [total, *phases.values(), *backends.values()]:
        assert type(count) is int
        assert 0 <= count < 2**64
    assert total == sum(phases.values()) == sum(backends.values())
    assert phases["routing"] == phases["approximate"] == 0
    assert backends["unknown"] == 0
    assert backends["sparse_fp32"] == total


@pytest.mark.parametrize("index_name", ["sindi", "sindi_v2"])
@pytest.mark.parametrize("use_reorder", [False, True])
def test_sindi_knn_search_with_statistics(index_name, use_reorder):
    """Expose one statistics JSON result per sparse CSR query."""
    index = pyvsag.Index(
        index_name,
        json.dumps(
            {
                "dim": 16,
                "dtype": "sparse",
                "metric_type": "ip",
                "index_param": {
                    "use_reorder": use_reorder,
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
            index_name: {
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
    np.testing.assert_array_equal(ids, legacy_ids)
    np.testing.assert_allclose(distances, legacy_distances)
    np.testing.assert_array_equal(ids, [[0, 1], [2, 0]])
    np.testing.assert_allclose(distances, [[-0.25, 0.2], [-0.17, 0.55]], atol=1e-6)
    assert len(statistics_json) == 2
    for value, candidate_count in zip(statistics_json, [3, 2]):
        assert isinstance(value, str)
        statistics = json.loads(value)
        assert_distance_statistics(statistics)
        # PR #2873 intentionally leaves partial statistics to keep posting scans fast.
        assert statistics["complete"] is False
        assert statistics["distance_evaluations_by_phase"]["rerank"] == (
            candidate_count if use_reorder else 0
        )

    # An absent term performs no approximate work and can still be complete.
    ids, distances, statistics_json = index.knn_search_with_statistics(
        np.array([0, 1], dtype=np.uint32),
        np.array([15], dtype=np.uint32),
        np.array([1.0], dtype=np.float32),
        2,
        search_parameters,
    )
    assert ids.shape == distances.shape == (1, 2)
    assert len(statistics_json) == 1
    statistics = json.loads(statistics_json[0])
    assert_distance_statistics(statistics)
    assert statistics["distance_evaluations"] == 0
    assert statistics["complete"] is True
    np.testing.assert_array_equal(ids, [[-1, -1]])
    np.testing.assert_array_equal(distances, [[-1.0, -1.0]])
