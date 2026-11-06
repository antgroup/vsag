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

"""Deterministic regressions for recall calculation."""

import numpy as np
import pytest

from conftest import TestDataset as Dataset, calculate_recall, verify_knn_search


def test_other_query_neighbors_do_not_count():
    assert calculate_recall(np.array([30, 40]), np.array([10, 20]), 2) == 0.0


def test_duplicate_ids_count_once():
    assert calculate_recall(np.array([10, 10]), np.array([10, 20]), 2) == 0.5
    assert calculate_recall(np.array([10, 20]), np.array([10, 10]), 2) == 0.5


def test_padding_does_not_match():
    assert calculate_recall(np.array([10, -1]), np.array([10, -1]), 2, padding_id=-1) == 0.5
    assert calculate_recall(np.array([-1, -1]), np.array([-1, -1]), 2, padding_id=-1) == 0.0


def test_negative_one_is_a_valid_id_by_default():
    assert calculate_recall(np.array([-1, 10]), np.array([-1, 10]), 2) == 1.0
    assert calculate_recall(np.array([-1]), np.array([-1]), 1) == 1.0


def test_explicit_padding_id_is_not_hardcoded():
    assert calculate_recall(np.array([-1, -99]), np.array([-1, -99]), 2,
                            padding_id=-99) == 0.5


def test_only_topk_results_and_ground_truth_count():
    assert calculate_recall(np.array([30, 10]), np.array([10, 30]), 1) == 0.0


def test_short_and_empty_results():
    assert calculate_recall(np.array([10]), np.array([10, 20]), 2) == 0.5
    assert calculate_recall(np.array([], dtype=np.int64), np.array([10, 20]), 2) == 0.0


@pytest.mark.parametrize("results,gt_ids", [([[10]], [10]), ([10], [[10]]), (10, [10])])
def test_recall_requires_one_query(results, gt_ids):
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_recall(np.asarray(results), np.asarray(gt_ids), 1)


@pytest.mark.parametrize("topk", [0, -1, 1.5])
def test_recall_requires_positive_integer_topk(topk):
    with pytest.raises(ValueError, match="positive integer"):
        calculate_recall(np.array([10]), np.array([10]), topk)


def test_ground_truth_maps_external_ids():
    dataset = Dataset(num_vectors=3, dim=2, topk=2)
    dataset.ids = np.array([101, 700, 9001], dtype=np.int64)
    vectors = np.array([[0, 0], [3, 0], [10, 0]], dtype=np.float32)
    queries = np.array([[9, 0], [1, 0]], dtype=np.float32)
    ids, _ = dataset.cal_gt_knn_results(vectors, queries, k=2)
    np.testing.assert_array_equal(ids, [[9001, 700], [101, 700]])


def test_verify_search_uses_each_query_ground_truth():
    dataset = Dataset(num_vectors=2, dim=1, topk=1)
    dataset.query_vectors_count = 2
    dataset.query_vectors = np.array([0, 1], dtype=np.float32)
    dataset.gt_ids = np.array([[101], [700]], dtype=np.int64)

    class WrongQueryIndex:
        def knn_search(self, vector, k, parameters):
            # Every answer belongs to the other query: aggregate GT would pass.
            other_id = 700 if vector[0] == 0 else 101
            return np.array([other_id]), np.array([0.0])

    with pytest.raises(AssertionError):
        verify_knn_search(WrongQueryIndex(), dataset, "{}", topk=1, expect_recall=0.9)
