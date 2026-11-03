# Copyright 2026 The VSAG Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Deterministic Python/native boundary regressions, not backend normalization."""
import json
import os
import sys

import numpy as np
import pytest
import pyvsag

# include/vsag/errors.h; parser/serialization errors deliberately do not pin codes.
INVALID_ARGUMENT = 3
INTERNAL_ERROR = 2
INDEX_NOT_EMPTY = 6
UNSUPPORTED_INDEX_OPERATION = 8


def dense_index(name, extra=None):
    parameters = {"dtype": "float32", "metric_type": "l2", "dim": 2}
    if name == "hgraph":
        parameters["index_param"] = {"max_degree": 8, "ef_construction": 32,
                                     "base_quantization_type": "fp32"}
    if extra is not None:
        parameters.setdefault("index_param", {}).update(extra)
    return pyvsag.Index(name, json.dumps(parameters))


def search_parameters(name):
    return json.dumps({"hgraph": {"ef_search": 32}}) if name == "hgraph" else "{}"


def built_dense(name):
    index = dense_index(name)
    vectors = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    assert index.build(vectors, np.array([10, 20, 30], dtype=np.int64), 3, 2) == []
    return index


def assert_core_error(operation, call, code=None):
    with pytest.raises(pyvsag.VsagError) as caught:
        call()
    error = caught.value
    assert isinstance(error, RuntimeError)
    assert type(error.code) is int
    assert error.code > 0
    assert error.operation == operation
    assert str(error)
    if code is not None:
        assert error.code == code
    return error


def assert_dense_arrays(result, expected_ids, expected_distances):
    ids, distances = result[:2]
    assert ids.dtype == np.dtype(np.int64)
    assert distances.dtype == np.dtype(np.float32)
    assert ids.shape == distances.shape == (len(expected_ids),)
    assert ids.flags.c_contiguous and distances.flags.c_contiguous
    if len(expected_ids):
        assert ids.strides == (8,)
        assert distances.strides == (4,)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-6)


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_dense_failed_ids_and_zero_row_batches(name):
    index = dense_index(name)
    empty = np.empty((0, 2), dtype=np.float32)
    empty_ids = np.empty(0, dtype=np.int64)
    # Public core Build/Add explicitly accept a zero-row batch.
    assert index.build(empty, empty_ids, 0, 2) == []
    assert index.add(empty, empty_ids, 0, 2) == []
    vectors = np.array([[0, 0], [1, 0], [1, 0]], dtype=np.float32)
    # Equal duplicate vectors make retained values independent of scheduling.
    failed = index.build(vectors, np.array([10, 20, 20], dtype=np.int64), 3, 2)
    assert isinstance(failed, list)
    assert sorted(failed) == [20]
    assert index.get_num_elements() == 2
    failed = index.add(np.array([[1, 0], [3, 0], [3, 0]], dtype=np.float32),
                       np.array([20, 30, 30], dtype=np.int64), 3, 2)
    assert isinstance(failed, list)
    assert sorted(failed) == [20, 30]
    assert index.add(np.array([[5, 0]], dtype=np.float32),
                     np.array([40], dtype=np.int64), 1, 2) == []
    assert index.get_num_elements() == 4
    assert index.build(empty, empty_ids, 0, 2) == []
    assert index.add(empty, empty_ids, 0, 2) == []


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_dense_actual_count_and_statistics(name):
    index = built_dense(name)
    query = np.zeros(2, dtype=np.float32)
    parameters = search_parameters(name)
    for method in ["knn_search", "knn_search_with_statistics"]:
        result = getattr(index, method)(query, 8, parameters)
        assert_dense_arrays(result, [10, 20, 30], [0, 1, 9])
        if method.endswith("with_statistics"):
            assert isinstance(result[2], str)
            assert isinstance(json.loads(result[2]), dict)
    for method in ["range_search", "range_search_with_statistics"]:
        assert_dense_arrays(getattr(index, method)(query, 2.0, parameters), [10, 20], [0, 1])
        assert_dense_arrays(getattr(index, method)(np.array([100, 0], dtype=np.float32),
                                                    0.01, parameters), [], [])


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_empty_dense_search_bypasses_backend_validation(name):
    index = dense_index(name)
    query = np.zeros(2, dtype=np.float32)
    for method in ["knn_search", "knn_search_with_statistics"]:
        # k=0 fails on populated HG/BF; public empty-index search returns early.
        assert_dense_arrays(getattr(index, method)(query, 0, search_parameters(name)), [], [])
    for method in ["range_search", "range_search_with_statistics"]:
        assert_dense_arrays(getattr(index, method)(query, -1.0, search_parameters(name)), [], [])


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_dense_expected_errors_have_public_operation(name):
    index = built_dense(name)
    query = np.zeros(2, dtype=np.float32)
    parameters = search_parameters(name)
    for method in ["knn_search", "knn_search_with_statistics"]:
        assert_core_error(method, lambda: getattr(index, method)(query, 0, parameters),
                          INVALID_ARGUMENT)
        assert_core_error(method, lambda: getattr(index, method)(np.zeros(3, dtype=np.float32),
                                                                 1, parameters), INVALID_ARGUMENT)
    for method in ["range_search", "range_search_with_statistics"]:
        assert_core_error(method, lambda: getattr(index, method)(query, -1.0, parameters),
                          INVALID_ARGUMENT)
    # Parser-specific code may vary, but ordinary/stats paths must agree.
    ordinary = assert_core_error("knn_search", lambda: index.knn_search(query, 1, "{"))
    stats = assert_core_error("knn_search_with_statistics",
                              lambda: index.knn_search_with_statistics(query, 1, "{"))
    assert ordinary.code == stats.code
    ordinary = assert_core_error("range_search", lambda: index.range_search(query, 1.0, "{"))
    stats = assert_core_error("range_search_with_statistics",
                              lambda: index.range_search_with_statistics(query, 1.0, "{"))
    assert ordinary.code == stats.code
    assert_core_error("add", lambda: index.add(np.zeros((1, 3), dtype=np.float32),
                                               np.array([40], dtype=np.int64), 1, 3),
                      INVALID_ARGUMENT)


def test_build_twice_hgraph_and_minmax_core_errors():
    index = built_dense("hgraph")
    assert_core_error("build", lambda: index.build(np.zeros((1, 2), dtype=np.float32),
                                                   np.array([40], dtype=np.int64), 1, 2),
                      INVALID_ARGUMENT)
    assert index.get_min_max_id() == (10, 30)
    assert_core_error("get_min_max_id", dense_index("hgraph").get_min_max_id, INTERNAL_ERROR)
    assert_core_error("get_min_max_id", built_dense("brute_force").get_min_max_id,
                      UNSUPPORTED_INDEX_OPERATION)


def test_established_unsupported_remove_configuration():
    # BF explicitly rejects removal with attribute filtering enabled.
    index = dense_index("brute_force", {"use_attribute_filter": True})
    assert index.build(np.zeros((1, 2), dtype=np.float32),
                       np.array([10], dtype=np.int64), 1, 2) == []
    assert_core_error("remove", lambda: index.remove(np.array([10], dtype=np.int64)),
                      INVALID_ARGUMENT)


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_strided_ids_are_controlled_copies(name):
    index = dense_index(name)
    ids = np.array([10, 999, 20, 999, 30, 999], dtype=np.int64)[::2]
    assert not ids.flags.c_contiguous
    vectors = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    assert index.build(vectors, ids, 3, 2) == []
    np.testing.assert_array_equal(ids, [10, 20, 30])
    assert_dense_arrays(index.knn_search(vectors[0], 8, search_parameters(name)),
                        [10, 20, 30], [0, 1, 9])
    remove_ids = np.array([10, 999, 30, 999], dtype=np.int64)[::2]
    assert index.remove(remove_ids) == 2
    assert index.get_num_elements() == 1
    assert_dense_arrays(index.knn_search(vectors[0], 8, search_parameters(name)), [20], [1])
    add_ids = np.array([40, 999, 50, 999], dtype=np.int64)[::2]
    # Flat contiguous dense buffers remain accepted; strided IDs are copied.
    assert index.add(np.array([5, 0, 7, 0], dtype=np.float32), add_ids, 2, 2) == []
    assert_dense_arrays(index.knn_search(vectors[0], 8, search_parameters(name)),
                        [20, 40, 50], [1, 25, 49])


@pytest.mark.parametrize("method", ["build", "add"])
@pytest.mark.parametrize("case", ["dtype", "layout", "shape", "rank", "ids_shape", "ids_length"])
def test_dense_batch_validation(method, case):
    index = dense_index("brute_force")
    vectors = np.zeros((3, 2), dtype=np.float32)
    ids = np.arange(3, dtype=np.int64)
    if case == "dtype":
        vectors = vectors.astype(np.float64)
    elif case == "layout":
        vectors = np.zeros((3, 4), dtype=np.float32)[:, ::2]
    elif case == "shape":
        vectors = np.zeros((2, 3), dtype=np.float32)
    elif case == "rank":
        vectors = vectors.reshape(1, 3, 2)
    elif case == "ids_shape":
        ids = ids.reshape(3, 1)
    elif case == "ids_length":
        ids = ids[:2]
    with pytest.raises(ValueError):
        getattr(index, method)(vectors, ids, 3, 2)


@pytest.mark.parametrize("method", ["build", "add"])
def test_dense_count_product_overflow_before_allocation(method):
    index = dense_index("brute_force")
    with pytest.raises(ValueError):
        getattr(index, method)(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64),
                               sys.maxsize, 2)


@pytest.mark.parametrize("method", ["knn_search", "knn_search_with_statistics",
                                    "range_search", "range_search_with_statistics"])
@pytest.mark.parametrize("case", ["dtype", "layout", "shape"])
def test_dense_query_validation(method, case):
    index = built_dense("brute_force")
    query = np.zeros(2, dtype=np.float32)
    if case == "dtype":
        query = query.astype(np.float64)
    elif case == "layout":
        query = np.zeros(4, dtype=np.float32)[::2]
    else:
        query = query.reshape(1, 2)
    with pytest.raises(ValueError):
        getattr(index, method)(query, 1, "{}")


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_dense_save_load_and_file_errors(name, tmp_path):
    index = built_dense(name)
    path = tmp_path / "index.bin"
    assert index.save(str(path)) is None
    restored = dense_index(name)
    assert restored.load(str(path)) is None
    assert restored.get_num_elements() == 3
    assert_dense_arrays(restored.knn_search(np.zeros(2, dtype=np.float32), 8,
                                            search_parameters(name)), [10, 20, 30], [0, 1, 9])
    assert_core_error("load", lambda: restored.load(str(path)), INDEX_NOT_EMPTY)
    with pytest.raises(OSError):
        dense_index(name).load(str(tmp_path / "missing.bin"))
    with pytest.raises(OSError):
        index.save(str(tmp_path / "missing_directory" / "index.bin"))
    # Truncate real serialization rather than inventing unsafe legacy count fields.
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(path.read_bytes()[:4])
    assert_core_error("load", lambda: dense_index(name).load(str(truncated)))
    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"")
    assert_core_error("load", lambda: dense_index(name).load(str(corrupt)))


@pytest.mark.skipif(not sys.platform.startswith("linux") or not os.path.exists("/dev/full"),
                    reason="requires Linux /dev/full")
def test_save_detects_final_flush_failure():
    # Tiny serialization fits in stream buffer, so ENOSPC appears on final flush.
    with pytest.raises(OSError):
        built_dense("brute_force").save("/dev/full")


def sindi_index():
    return pyvsag.Index("sindi", json.dumps({
        "dim": 16, "dtype": "sparse", "metric_type": "ip",
        "index_param": {"use_reorder": False, "doc_prune_ratio": 0.0,
                        "window_size": 10000, "term_id_limit": 16},
    }))


def sparse_parameters():
    return json.dumps({"sindi": {"n_candidate": 3, "query_prune_ratio": 0.0,
                                  "term_prune_ratio": 0.0}})


def strided(values, dtype):
    result = np.empty(len(values) * 2, dtype=dtype)
    result[::2] = values
    result[1::2] = 999
    return result[::2]


def build_sindi(index):
    # Each field needs a dtype conversion and a contiguous copy.
    return index.build(strided([0, 2, 4, 6], np.int64),
                       strided([0, 1, 0, 2, 1, 3], np.int64),
                       strided([1.0, 0.5, 0.8, 0.7, 0.9, 0.6], np.float64),
                       strided([10, 20, 30], np.int32))


def test_sparse_failed_ids_and_zero_rows():
    empty = sindi_index()
    assert empty.build(np.array([0], dtype=np.uint32), np.empty(0, dtype=np.uint32),
                       np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)) == []
    assert empty.get_num_elements() == 0
    assert build_sindi(empty) == []
    index = sindi_index()
    # Duplicate rows have equal values; insertion order cannot change the oracle.
    failed = index.build(np.array([0, 1, 2, 3], dtype=np.uint32),
                         np.array([0, 1, 1], dtype=np.uint32),
                         np.ones(3, dtype=np.float32), np.array([10, 20, 20], dtype=np.int64))
    assert isinstance(failed, list)
    assert sorted(failed) == [20]
    assert index.get_num_elements() == 2
    # Mutable SINDI Build appends via add(); only immutable mode is build-once.
    assert sorted(build_sindi(index)) == [10, 20]
    assert index.get_num_elements() == 3


@pytest.mark.parametrize("method", ["knn_search", "knn_search_with_statistics"])
def test_sparse_rectangular_padding_strides_and_controlled_copies(method):
    index = sindi_index()
    assert build_sindi(index) == []
    # Rows match three docs, two docs, and no posting list respectively.
    pointers = strided([0, 2, 4, 5], np.int64)
    indices = strided([0, 1, 1, 3, 15], np.int64)
    values = strided([1.0, 0.5, 0.9, 0.6, 1.0], np.float64)
    result = getattr(index, method)(pointers, indices, values, 5, sparse_parameters())
    ids, distances = result[:2]
    assert ids.shape == distances.shape == (3, 5)
    assert ids.dtype == np.dtype(np.int64)
    assert distances.dtype == np.dtype(np.float32)
    assert ids.strides == (40, 8)
    assert distances.strides == (20, 4)
    assert ids.flags.c_contiguous and distances.flags.c_contiguous
    np.testing.assert_array_equal(ids, [[10, 20, 30, -1, -1],
                                       [30, 10, -1, -1, -1], [-1, -1, -1, -1, -1]])
    np.testing.assert_allclose(distances[0, :3], [-0.25, 0.2, 0.55], atol=1e-6)
    np.testing.assert_allclose(distances[1, :2], [-0.17, 0.55], atol=1e-6)
    assert np.isposinf(distances[0, 3:]).all()
    assert np.isposinf(distances[1, 2:]).all()
    assert np.isposinf(distances[2]).all()
    np.testing.assert_array_equal(pointers, [0, 2, 4, 5])
    np.testing.assert_array_equal(indices, [0, 1, 1, 3, 15])
    if method.endswith("with_statistics"):
        assert len(result[2]) == 3
        for statistics in result[2]:
            assert isinstance(json.loads(statistics), dict)


@pytest.mark.parametrize("method", ["knn_search", "knn_search_with_statistics"])
def test_sparse_expected_failures_are_not_padding(method):
    index = sindi_index()
    assert build_sindi(index) == []
    search = getattr(index, method)
    assert_core_error(method, lambda: search(np.array([0, 0], dtype=np.uint32),
                                             np.empty(0, dtype=np.uint32),
                                             np.empty(0, dtype=np.float32), 2, sparse_parameters()),
                      INVALID_ARGUMENT)
    assert_core_error(method, lambda: search(np.array([0, 1], dtype=np.uint32),
                                             np.array([0], dtype=np.uint32),
                                             np.ones(1, dtype=np.float32), 2, "{"))


@pytest.mark.parametrize("pointers,indices,values", [
    ([], [], []), ([1, 1], [0], [1.0]), ([0, 2, 1], [0], [1.0]),
    ([0, 2], [0], [1.0]), ([0, 1], [0], []),
])
def test_sparse_csr_validation(pointers, indices, values):
    index = sindi_index()
    with pytest.raises(ValueError):
        index.build(np.array(pointers, dtype=np.uint32), np.array(indices, dtype=np.uint32),
                    np.array(values, dtype=np.float32), np.array([10], dtype=np.int64))


@pytest.mark.parametrize("name", ["hgraph", "brute_force"])
def test_dense_singlequery_preserves_legitimate_negative_one_id(name):
    index = dense_index(name)
    vectors = np.array([[0, 0], [1, 0]], dtype=np.float32)
    assert index.build(vectors, np.array([-1, 20], dtype=np.int64), 2, 2) == []
    parameters = search_parameters(name)
    # Native batch HG reserves -1; these public calls are SINGLEQUERY, not batch.
    for method in ["knn_search", "knn_search_with_statistics"]:
        assert_dense_arrays(getattr(index, method)(vectors[0], 5, parameters), [-1, 20], [0, 1])
    for method in ["range_search", "range_search_with_statistics"]:
        assert_dense_arrays(getattr(index, method)(vectors[0], 0.5, parameters), [-1], [0])


@pytest.mark.parametrize("method", ["knn_search", "knn_search_with_statistics"])
def test_sparse_legitimate_negative_one_is_distinguished_from_padding(method):
    index = sindi_index()
    assert index.build(np.array([0, 1], dtype=np.uint32),
                       np.array([0], dtype=np.uint32), np.ones(1, dtype=np.float32),
                       np.array([-1], dtype=np.int64)) == []
    # LabelTable accepts -1; distinguish a real result by its finite distance.
    result = getattr(index, method)(np.array([0, 1], dtype=np.uint32),
                                    np.array([0], dtype=np.uint32),
                                    np.ones(1, dtype=np.float32), 3, sparse_parameters())
    np.testing.assert_array_equal(result[0], [[-1, -1, -1]])
    np.testing.assert_allclose(result[1][0, :1], [0.0], atol=1e-6)
    assert np.isfinite(result[1][0, 0])
    assert np.isposinf(result[1][0, 1:]).all()


@pytest.mark.parametrize("method", ["knn_search", "knn_search_with_statistics"])
def test_sparse_empty_index_and_zero_query_rows(method):
    index = sindi_index()
    # Public core empty-index check returns before SINDI query validation.
    result = getattr(index, method)(np.array([0, 0], dtype=np.uint32),
                                    np.empty(0, dtype=np.uint32),
                                    np.empty(0, dtype=np.float32), 3, sparse_parameters())
    assert result[0].shape == result[1].shape == (1, 3)
    np.testing.assert_array_equal(result[0], [[-1, -1, -1]])
    assert np.isposinf(result[1]).all()
    assert build_sindi(index) == []
    # CSR [0] means no queries, not one invalid empty sparse vector.
    result = getattr(index, method)(np.array([0], dtype=np.uint32),
                                    np.empty(0, dtype=np.uint32),
                                    np.empty(0, dtype=np.float32), 3, sparse_parameters())
    assert result[0].shape == result[1].shape == (0, 3)
    assert result[0].dtype == np.dtype(np.int64)
    assert result[1].dtype == np.dtype(np.float32)
    if method.endswith("with_statistics"):
        assert list(result[2]) == []
