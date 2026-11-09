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

"""Pytest tests for bit-packed binary vectors with the Hamming metric."""

import json

import numpy as np
import pytest

import pyvsag

SEARCH_PARAMS = json.dumps({"hgraph": {"ef_search": 200}})


def _params(dim, index_param=None):
    return json.dumps(
        {
            "dtype": "binary",
            "metric_type": "hamming",
            "dim": dim,
            "index_param": index_param or {},
        }
    )


def _hgraph_params(dim):
    return _params(dim, {"max_degree": 24, "ef_construction": 150})


def _random_bits(rng, rows, dim):
    return rng.integers(0, 2, size=(rows, dim), dtype=np.uint8).astype(bool)


def _pack(bits):
    # numpy.packbits packs MSB-first by default: logical bit i -> byte i // 8, mask 0x80 >> i % 8.
    return np.packbits(bits, axis=-1)


def _hamming(bits_a, bits_b):
    return np.count_nonzero(bits_a != bits_b, axis=-1)


@pytest.mark.parametrize("index_name", ["brute_force", "hgraph"])
@pytest.mark.parametrize("dim", [1, 7, 8, 9, 63, 64, 65, 127, 128, 300])
def test_binary_knn_distances_are_exact(index_name, dim):
    rng = np.random.default_rng(dim)
    num = 400
    bits = _random_bits(rng, num, dim)
    ids = np.arange(num, dtype=np.int64) * 7 + 3
    params = _params(dim) if index_name == "brute_force" else _hgraph_params(dim)
    index = pyvsag.Index(index_name, params)
    index.build(_pack(bits), ids, num, dim)
    assert index.get_num_elements() == num

    k = 10
    for _ in range(5):
        query_bits = _random_bits(rng, 1, dim)[0]
        expected = _hamming(bits, query_bits)
        result_ids, result_dists = index.knn_search(_pack(query_bits), k, SEARCH_PARAMS)
        positions = (result_ids - 3) // 7
        # Every returned distance is the exact number of differing bits of its id.
        np.testing.assert_array_equal(result_dists, expected[positions].astype(np.float32))
        kth = np.sort(expected)[k - 1]
        if index_name == "brute_force":
            np.testing.assert_array_equal(np.sort(result_dists), np.sort(expected)[:k])
        else:
            assert np.mean(result_dists <= kth) >= 0.8


@pytest.mark.parametrize("dim", [9, 65, 256])
def test_binary_bruteforce_range_and_distance_by_id(dim):
    rng = np.random.default_rng(100 + dim)
    num = 300
    bits = _random_bits(rng, num, dim)
    ids = np.arange(num, dtype=np.int64)
    index = pyvsag.Index("brute_force", _params(dim))
    index.build(_pack(bits), ids, num, dim)

    query_bits = bits[17] ^ (rng.random(dim) < 0.05)
    expected = _hamming(bits, query_bits)
    radius = float(np.sort(expected)[20])
    range_ids, range_dists = index.range_search(_pack(query_bits), radius, "{}")
    assert set(range_ids.tolist()) == set(np.nonzero(expected <= radius)[0].tolist())
    np.testing.assert_array_equal(range_dists, expected[range_ids].astype(np.float32))

    probe = np.array([0, 17, num - 1, 10**9], dtype=np.int64)
    distances = index.calc_distances_by_id(_pack(query_bits), probe)
    np.testing.assert_array_equal(distances[:3], expected[probe[:3]].astype(np.float32))
    assert distances[3] < 0


def test_binary_inputs_accept_flat_buffers_and_ignore_padding():
    dim = 13  # two bytes per vector, three padding bits
    rng = np.random.default_rng(5)
    bits = _random_bits(rng, 50, dim)
    packed = _pack(bits)
    ids = np.arange(50, dtype=np.int64)

    index = pyvsag.Index("brute_force", _params(dim))
    index.build(packed[:30].reshape(-1), ids[:30], 30, dim)  # flat buffer
    index.add(packed[30:], ids[30:], 20, dim)  # (rows, ceil(dim / 8)) matrix
    assert index.get_num_elements() == 50

    query = _pack(bits[7])
    dirty = query.copy()
    dirty[-1] |= 0b00000111  # garbage in the padding bits
    clean_result = index.knn_search(query, 5, "{}")
    dirty_result = index.knn_search(dirty, 5, "{}")
    np.testing.assert_array_equal(clean_result[1], dirty_result[1])
    assert clean_result[1][0] == 0
    np.testing.assert_array_equal(
        index.calc_distances_by_id(dirty, ids), _hamming(bits, bits[7]).astype(np.float32)
    )


def test_binary_index_save_and_load(tmp_path):
    dim = 200
    rng = np.random.default_rng(9)
    bits = _random_bits(rng, 300, dim)
    ids = np.arange(300, dtype=np.int64)
    index = pyvsag.Index("hgraph", _hgraph_params(dim))
    index.build(_pack(bits), ids, 300, dim)
    path = str(tmp_path / "binary_hgraph.index")
    index.save(path)

    loaded = pyvsag.Index("hgraph", _hgraph_params(dim))
    loaded.load(path)
    query = _pack(bits[3])
    before = index.knn_search(query, 10, SEARCH_PARAMS)
    after = loaded.knn_search(query, 10, SEARCH_PARAMS)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize(
    "bad_vectors",
    [
        np.zeros((10, 2), dtype=np.float32),
        np.zeros((10, 2), dtype=np.int8),
        np.zeros((10, 2), dtype=np.uint16),
        np.zeros((10, 2), dtype=bool),
    ],
)
def test_binary_rejects_wrong_dtype(bad_vectors):
    index = pyvsag.Index("brute_force", _params(13))
    with pytest.raises(ValueError, match="numpy.uint8"):
        index.build(bad_vectors, np.arange(10, dtype=np.int64), 10, 13)


@pytest.mark.parametrize(
    "bad_vectors",
    [
        np.zeros((10, 1), dtype=np.uint8),  # one byte per row, 13 bits need two
        np.zeros((10, 3), dtype=np.uint8),
        np.zeros((9, 2), dtype=np.uint8),
        np.zeros(19, dtype=np.uint8),
        np.zeros((10, 2, 1), dtype=np.uint8),
    ],
)
def test_binary_rejects_wrong_shape(bad_vectors):
    index = pyvsag.Index("hgraph", _hgraph_params(13))
    with pytest.raises(ValueError):
        index.build(bad_vectors, np.arange(10, dtype=np.int64), 10, 13)


def test_binary_rejects_non_contiguous_arrays():
    dim = 16
    index = pyvsag.Index("brute_force", _params(dim))
    wide = np.zeros((10, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        index.build(wide[:, ::2], np.arange(10, dtype=np.int64), 10, dim)
    with pytest.raises(ValueError, match="C-contiguous"):
        index.build(np.zeros((2, 10), dtype=np.uint8).T, np.arange(10, dtype=np.int64), 10, dim)
    # The documented safe conversion works.
    index.build(np.ascontiguousarray(wide[:, ::2]), np.arange(10, dtype=np.int64), 10, dim)
    assert index.get_num_elements() == 10


def test_binary_rejects_wrong_query_length_and_dtype():
    dim = 65  # nine bytes
    rng = np.random.default_rng(1)
    index = pyvsag.Index("hgraph", _hgraph_params(dim))
    index.build(_pack(_random_bits(rng, 20, dim)), np.arange(20, dtype=np.int64), 20, dim)
    for query in (np.zeros(8, dtype=np.uint8), np.zeros(10, dtype=np.uint8)):
        with pytest.raises(ValueError, match="bytes"):
            index.knn_search(query, 3, SEARCH_PARAMS)
        with pytest.raises(ValueError):
            index.range_search(query, 3.0, SEARCH_PARAMS)
        with pytest.raises(ValueError):
            index.calc_distances_by_id(query, np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="numpy.uint8"):
        index.knn_search(np.zeros(9, dtype=np.float32), 3, SEARCH_PARAMS)
    with pytest.raises(ValueError):
        index.calc_distances_by_id([0] * 9, np.arange(3, dtype=np.int64))


def test_binary_requires_hamming_and_supported_index():
    with pytest.raises(RuntimeError):
        pyvsag.Index(
            "brute_force", json.dumps({"dtype": "binary", "metric_type": "l2", "dim": 8})
        )
    with pytest.raises(RuntimeError):
        pyvsag.Index(
            "hgraph", json.dumps({"dtype": "float32", "metric_type": "hamming", "dim": 8})
        )
    with pytest.raises(RuntimeError):
        pyvsag.Index("ivf", _params(64))
