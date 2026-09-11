# Copyright 2024-present the vsag project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Distance binding safety; requires freshly rebuilt pyvsag and matching core."""

import json

import numpy as np
import pyvsag
import pytest


@pytest.fixture(params=["calc_distances_by_id", "cal_distance_by_id"])
def distance_method(request):
    return request.param


@pytest.fixture
def index():
    result = pyvsag.Index("hgraph", json.dumps({
        "dtype": "float32", "metric_type": "l2", "dim": 4,
        "index_param": {"base_quantization_type": "fp32", "max_degree": 16,
                        "ef_construction": 50},
    }))
    result.build(np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32),
                 np.array([10, 20], dtype=np.int64), 2, 4)
    return result


@pytest.mark.parametrize("length", [0, 3, 5])
@pytest.mark.parametrize("ids", [[10], [999]])
def test_wrong_query_length_propagates_cpp_error(index, distance_method, length, ids):
    # Missing candidates must not conceal an invalid query or return all -1.
    with pytest.raises(RuntimeError, match=r"(?i)(dimension|dim|query|vector)"):
        getattr(index, distance_method)(np.zeros(length, dtype=np.float32),
                                        np.array(ids, dtype=np.int64))


def test_missing_ids_and_legacy_shape(index, distance_method):
    result = getattr(index, distance_method)(
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([10, 999, 20, 10], dtype=np.int64))
    assert result.shape == (4,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [0, -1, 20, 0], atol=1e-5)
    empty = getattr(index, distance_method)(np.ones(4, dtype=np.float32),
                                            np.array([], dtype=np.int64))
    assert empty.shape == (0,)
    assert empty.dtype == np.float32


@pytest.mark.parametrize("reverse", [False, True])
def test_strided_query_and_ids_preserve_logical_order(index, distance_method, reverse):
    query = np.array([1, 99, 2, 99, 3, 99, 4, 99], dtype=np.float32)[::2]
    ids = np.array([10, 888, 999, 888, 20, 888, 10, 888], dtype=np.int64)[::2]
    if reverse:
        query, ids = query[::-1], ids[::-1]
    assert not query.flags.c_contiguous
    assert not ids.flags.c_contiguous
    method = getattr(index, distance_method)
    expected = method(np.ascontiguousarray(query), np.ascontiguousarray(ids))
    np.testing.assert_array_equal(method(query, ids), expected)


@pytest.mark.parametrize("argument", ["query", "ids"])
def test_rejects_multiple_dimensions(index, distance_method, argument):
    query = np.ones(4, dtype=np.float32)
    ids = np.array([10], dtype=np.int64)
    if argument == "query":
        query = query.reshape(1, 4)
    else:
        ids = ids.reshape(1, 1)
    with pytest.raises(ValueError, match="1-dimensional"):
        getattr(index, distance_method)(query, ids)


def test_unsupported_dense_representation_is_not_missing(distance_method):
    # SINDI accepts sparse queries, not this binding's dense query representation.
    sparse = pyvsag.Index("sindi", json.dumps({
        "dim": 16, "dtype": "sparse", "metric_type": "ip",
        "index_param": {"use_reorder": True, "doc_prune_ratio": 0.0,
                        "window_size": 10000, "term_id_limit": 16},
    }))
    sparse.build(np.array([0, 1], dtype=np.uint32), np.array([0], dtype=np.uint32),
                 np.array([1.0], dtype=np.float32), np.array([10], dtype=np.int64))
    with pytest.raises(RuntimeError, match=r"(?i)(sparse|representation|support)"):
        getattr(sparse, distance_method)(np.ones(16, dtype=np.float32),
                                         np.array([10, 999], dtype=np.int64))
