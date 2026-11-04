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

"""Pytest tests for bit-packed binary vectors and Hamming distance."""

import json

import numpy as np
import pyvsag
import pytest

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hamming_matrix(queries: np.ndarray, base: np.ndarray, dim: int) -> np.ndarray:
    """Exact Hamming distance with unused tail bits masked out."""
    bytes_per_vector = (dim + 7) // 8
    q = queries.reshape(-1, bytes_per_vector)
    b = base.reshape(-1, bytes_per_vector)
    xor = np.bitwise_xor(q[:, None, :], b[None, :, :])
    dists = POPCOUNT[xor].sum(axis=2, dtype=np.uint32)
    used_tail_bits = dim % 8
    if used_tail_bits:
        mask = (1 << used_tail_bits) - 1
        tail = POPCOUNT[(q[:, None, -1] ^ b[None, :, -1]) & ~mask & 0xFF]
        dists = dists - tail
    return dists


def index_params(index_type: str, dim: int) -> str:
    return json.dumps({"dtype": "binary", "metric_type": "hamming", "dim": dim})


def make_binary_data(num_elements: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bytes_per_vector = (dim + 7) // 8
    return rng.integers(0, 256, size=(num_elements, bytes_per_vector), dtype=np.uint8)


def test_bruteforce_binary_hamming_valid_input():
    dim = 137  # not a multiple of eight
    num_elements = 400
    base = make_binary_data(num_elements, dim, 11)
    ids = np.arange(num_elements, dtype=np.int64)
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    index.build(vectors=base, ids=ids, num_elements=num_elements, dim=dim)

    queries = make_binary_data(8, dim, 12)
    ground = hamming_matrix(queries, base, dim)
    for query, row in zip(queries, ground):
        result_ids, result_dists = index.knn_search(
            vector=query, k=10, parameters=json.dumps({})
        )
        expected = np.sort(row)[:10]
        np.testing.assert_array_equal(result_dists, expected.astype(np.float32))
        for result_id, result_dist in zip(result_ids, result_dists):
            assert row[result_id] == int(result_dist)

        threshold = float(np.sort(row)[49])
        range_ids, range_dists = index.range_search(
            point=query, threshold=threshold, parameters=json.dumps({})
        )
        expected_range = np.count_nonzero(row <= threshold)
        assert len(range_ids) == expected_range
        assert np.all(range_dists <= threshold)

        by_id = index.calc_distances_by_id(
            query=query, ids=np.array([0, 7, num_elements - 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            by_id, row[[0, 7, num_elements - 1]].astype(np.float32)
        )


def test_bruteforce_binary_flat_buffer_and_add():
    dim = 64
    num_elements = 200
    bytes_per_vector = dim // 8
    base = make_binary_data(num_elements, dim, 21)
    ids = np.arange(num_elements, dtype=np.int64)
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    index.build(
        vectors=base.reshape(-1), ids=ids, num_elements=num_elements, dim=dim
    )

    extra = make_binary_data(20, dim, 22)
    extra_ids = np.arange(num_elements, num_elements + 20, dtype=np.int64)
    index.add(
        vectors=extra.reshape(-1), ids=extra_ids, num_elements=20, dim=dim
    )
    assert index.get_num_elements() == num_elements + 20

    query = make_binary_data(1, dim, 23)[0]
    all_base = np.concatenate([base, extra], axis=0)
    row = hamming_matrix(query.reshape(1, -1), all_base, dim)[0]
    result_ids, result_dists = index.knn_search(
        vector=query, k=5, parameters=json.dumps({})
    )
    np.testing.assert_array_equal(result_dists, np.sort(row)[:5].astype(np.float32))
    assert bytes_per_vector == base.shape[1]


def test_binary_unused_tail_bits_are_ignored():
    dim = 9  # two bytes, only the low bit of the second byte is meaningful
    rng = np.random.default_rng(31)
    base = rng.integers(0, 256, size=(16, 2), dtype=np.uint8)
    ids = np.arange(base.shape[0], dtype=np.int64)
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    index.build(vectors=base, ids=ids, num_elements=base.shape[0], dim=dim)

    query = base[3].copy()
    query[1] ^= 0xFE  # only unused tail bits differ
    result_ids, result_dists = index.knn_search(
        vector=query, k=1, parameters=json.dumps({})
    )
    assert result_ids[0] == 3
    assert result_dists[0] == 0.0


@pytest.mark.parametrize("dim", [1, 7, 8, 9, 63, 64, 65, 127, 128])
def test_binary_non_aligned_dimensions(dim):
    num_elements = 32
    base = make_binary_data(num_elements, dim, dim + 100)
    ids = np.arange(num_elements, dtype=np.int64)
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    index.build(vectors=base, ids=ids, num_elements=num_elements, dim=dim)

    query = make_binary_data(1, dim, dim + 200)[0]
    row = hamming_matrix(query.reshape(1, -1), base, dim)[0]
    result_ids, result_dists = index.knn_search(
        vector=query, k=5, parameters=json.dumps({})
    )
    np.testing.assert_array_equal(result_dists, np.sort(row)[:5].astype(np.float32))


def test_binary_rejects_invalid_dtype():
    dim = 64
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    vectors = np.zeros((10, dim // 8), dtype=np.float32)
    with pytest.raises(RuntimeError):
        index.build(
            vectors=vectors,
            ids=np.arange(10, dtype=np.int64),
            num_elements=10,
            dim=dim,
        )


def test_binary_rejects_invalid_shape_and_query_length():
    dim = 65  # ceil(65 / 8) == 9 bytes
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    vectors = np.zeros((10, 8), dtype=np.uint8)  # one byte short
    with pytest.raises(RuntimeError):
        index.build(
            vectors=vectors,
            ids=np.arange(10, dtype=np.int64),
            num_elements=10,
            dim=dim,
        )

    valid = np.zeros((10, 9), dtype=np.uint8)
    index.build(
        vectors=valid,
        ids=np.arange(10, dtype=np.int64),
        num_elements=10,
        dim=dim,
    )
    with pytest.raises(RuntimeError):
        index.knn_search(vector=np.zeros(8, dtype=np.uint8), k=1, parameters="{}")
    with pytest.raises(RuntimeError):
        index.range_search(
            point=np.zeros(10, dtype=np.uint8), threshold=1.0, parameters="{}"
        )


def test_binary_rejects_non_contiguous_arrays():
    dim = 64
    index = pyvsag.Index("brute_force", index_params("brute_force", dim))
    contiguous = np.zeros((10, dim // 8), dtype=np.uint8)
    non_contiguous = np.asfortranarray(contiguous)
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    with pytest.raises(RuntimeError):
        index.build(
            vectors=non_contiguous,
            ids=np.arange(10, dtype=np.int64),
            num_elements=10,
            dim=dim,
        )


def test_binary_hgraph_recall_against_bruteforce():
    dim = 128
    num_elements = 600
    ids = np.arange(num_elements, dtype=np.int64)
    base = make_binary_data(num_elements, dim, 41)
    queries = make_binary_data(20, dim, 42)

    exact = pyvsag.Index("brute_force", index_params("brute_force", dim))
    exact.build(vectors=base, ids=ids, num_elements=num_elements, dim=dim)

    graph = pyvsag.Index("hgraph", index_params("hgraph", dim))
    graph.build(vectors=base, ids=ids, num_elements=num_elements, dim=dim)

    search_params = json.dumps({"hgraph": {"ef_search": 200}})
    recall_hits = 0
    for query in queries:
        exact_ids, _ = exact.knn_search(
            vector=query, k=10, parameters=json.dumps({})
        )
        graph_ids, graph_dists = graph.knn_search(
            vector=query, k=10, parameters=search_params
        )
        recall_hits += len(set(exact_ids.tolist()) & set(graph_ids.tolist()))
        row = hamming_matrix(query.reshape(1, -1), base, dim)[0]
        for result_id, result_dist in zip(graph_ids, graph_dists):
            assert row[result_id] == int(result_dist)

    recall = recall_hits / (len(queries) * 10)
    assert recall >= 0.8
