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

"""Tests for extended Python index operations"""

import numpy as np
import pyvsag
import pytest

from conftest import (
    TestDataset,
    create_index,
    build_index,
)


def _create_dense_dispatch_index(dtype: str, dim: int) -> pyvsag.Index:
    quantization = "fp16" if dtype == "float16" else "bf16"
    return create_index(
        "hgraph",
        f"""
        {{
            "dtype": "{dtype}",
            "metric_type": "l2",
            "dim": {dim},
            "index_param": {{
                "base_quantization_type": "{quantization}",
                "max_degree": 16,
                "ef_construction": 50,
                "alpha": 1.2
            }}
        }}
        """,
    )


def _float32_to_bfloat16_bits(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


class TestGetNumElements:
    """Tests for get_num_elements method"""

    def test_get_num_elements_after_build(self, dataset):
        """Test get_num_elements returns correct count after building"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        assert index.get_num_elements() == dataset.num_elements

    def test_get_num_elements_empty_index(self):
        """Test get_num_elements returns 0 for empty index"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)

        assert index.get_num_elements() == 0


class TestGetMemoryUsage:
    """Tests for get_memory_usage method"""

    def test_get_memory_usage_after_build(self, dataset):
        """Test get_memory_usage returns positive value after building"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        memory = index.get_memory_usage()
        assert memory > 0

    def test_get_memory_usage_empty_index(self):
        """Test get_memory_usage returns minimal value for empty index"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)

        memory = index.get_memory_usage()
        assert memory >= 0


class TestCheckIdExist:
    """Tests for check_id_exist method"""

    def test_check_id_exist_true(self, dataset):
        """Test check_id_exist returns True for existing ID"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        assert index.check_id_exist(0) == True
        assert index.check_id_exist(100) == True

    def test_check_id_exist_false(self, dataset):
        """Test check_id_exist returns False for non-existing ID"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        assert index.check_id_exist(999999) == False

    def test_check_id_exist_negative_id(self, dataset):
        """Test check_id_exist returns False for negative ID"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        assert index.check_id_exist(-1) == False


class TestGetMinMaxId:
    """Tests for get_min_max_id method"""

    def test_get_min_max_id_after_build(self, dataset):
        """Test get_min_max_id returns correct range after building"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        min_id, max_id = index.get_min_max_id()
        assert min_id == 0
        assert max_id == dataset.num_elements - 1


class TestAddVectors:
    """Tests for add method"""

    def test_add_vectors_to_empty_index(self, dataset_factory):
        """Test adding vectors to empty index"""
        dataset = dataset_factory(num_vectors=100, dim=64)

        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 64,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)

        initial_count = index.get_num_elements()
        assert initial_count == 0

        index.add(dataset.base_vectors, dataset.ids, dataset.num_elements, dataset.dim)

        assert index.get_num_elements() == dataset.num_elements

    def test_add_vectors_incrementally(self, dataset_factory):
        """Test adding vectors incrementally"""
        dataset = dataset_factory(num_vectors=100, dim=64)

        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 64,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)

        half = dataset.num_elements // 2

        vectors_1 = dataset.base_vectors[: half * dataset.dim]
        ids_1 = dataset.ids[:half]
        index.add(vectors_1, ids_1, half, dataset.dim)
        assert index.get_num_elements() == half

        vectors_2 = dataset.base_vectors[half * dataset.dim :]
        ids_2 = dataset.ids[half:]
        index.add(vectors_2, ids_2, dataset.num_elements - half, dataset.dim)
        assert index.get_num_elements() == dataset.num_elements


class TestDenseDtypeDispatch:
    """Tests for dense dtype-aware build/search/add bindings"""

    def test_build_accepts_2d_float16_matrix(self, dataset_factory):
        dataset = dataset_factory(num_vectors=64, dim=32)
        index = _create_dense_dispatch_index("float16", dataset.dim)

        vectors = dataset.base_vectors.reshape(dataset.num_elements, dataset.dim).astype(np.float16)
        index.build(vectors, dataset.ids, dataset.num_elements, dataset.dim)

        assert index.get_num_elements() == dataset.num_elements

    def test_add_accepts_2d_bfloat16_matrix(self, dataset_factory):
        dataset = dataset_factory(num_vectors=32, dim=16)
        index = _create_dense_dispatch_index("bfloat16", dataset.dim)

        vectors = _float32_to_bfloat16_bits(dataset.base_vectors).reshape(
            dataset.num_elements, dataset.dim
        )
        index.add(vectors, dataset.ids, dataset.num_elements, dataset.dim)

        assert index.get_num_elements() == dataset.num_elements

    def test_knn_search_accepts_float16_query(self, dataset_factory):
        dataset = dataset_factory(num_vectors=64, dim=32)
        index = _create_dense_dispatch_index("float16", dataset.dim)
        index.build(dataset.base_vectors.astype(np.float16), dataset.ids, dataset.num_elements, dataset.dim)

        query = dataset.query_vectors[: dataset.dim].astype(np.float16)
        ids, distances = index.knn_search(query, 5, '{"hgraph": {"ef_search": 20}}')

        assert ids.shape == (5,)
        assert distances.shape == (5,)

    def test_knn_search_accepts_bfloat16_query(self, dataset_factory):
        dataset = dataset_factory(num_vectors=64, dim=32)
        index = _create_dense_dispatch_index("bfloat16", dataset.dim)
        index.build(
            _float32_to_bfloat16_bits(dataset.base_vectors),
            dataset.ids,
            dataset.num_elements,
            dataset.dim,
        )

        query = _float32_to_bfloat16_bits(dataset.query_vectors[: dataset.dim])
        ids, distances = index.knn_search(query, 5, '{"hgraph": {"ef_search": 20}}')

        assert ids.shape == (5,)
        assert distances.shape == (5,)

    def test_range_search_accepts_float16_query(self, dataset_factory):
        dataset = dataset_factory(num_vectors=64, dim=32)
        index = _create_dense_dispatch_index("float16", dataset.dim)
        index.build(dataset.base_vectors.astype(np.float16), dataset.ids, dataset.num_elements, dataset.dim)

        query = dataset.query_vectors[: dataset.dim].astype(np.float16)
        ids, distances = index.range_search(query, 1e9, '{"hgraph": {"ef_search": 20}}')

        assert ids.ndim == 1
        assert distances.ndim == 1
        assert ids.shape == distances.shape

    def test_range_search_accepts_bfloat16_query(self, dataset_factory):
        dataset = dataset_factory(num_vectors=64, dim=32)
        index = _create_dense_dispatch_index("bfloat16", dataset.dim)
        index.build(
            _float32_to_bfloat16_bits(dataset.base_vectors),
            dataset.ids,
            dataset.num_elements,
            dataset.dim,
        )

        query = _float32_to_bfloat16_bits(dataset.query_vectors[: dataset.dim])
        ids, distances = index.range_search(query, 1e9, '{"hgraph": {"ef_search": 20}}')

        assert ids.ndim == 1
        assert distances.ndim == 1
        assert ids.shape == distances.shape

    @pytest.mark.parametrize(
        ("dtype", "vectors", "expected_error"),
        [
            ("float16", lambda dataset: dataset.base_vectors.astype(np.float32), "vectors must be numpy.float16 array"),
            (
                "bfloat16",
                lambda dataset: dataset.base_vectors.astype(np.float16),
                r"vectors must be numpy.uint16 \(raw bfloat16 bits\) array",
            ),
        ],
    )
    def test_add_rejects_wrong_dense_dtype(self, dataset_factory, dtype, vectors, expected_error):
        dataset = dataset_factory(num_vectors=16, dim=8)
        index = _create_dense_dispatch_index(dtype, dataset.dim)

        with pytest.raises(ValueError, match=expected_error):
            index.add(vectors(dataset), dataset.ids, dataset.num_elements, dataset.dim)

    @pytest.mark.parametrize(
        ("dtype", "vectors", "expected_error"),
        [
            ("float16", lambda dataset: dataset.base_vectors.astype(np.float32), "vectors must be numpy.float16 array"),
            (
                "bfloat16",
                lambda dataset: dataset.base_vectors.astype(np.float16),
                r"vectors must be numpy.uint16 \(raw bfloat16 bits\) array",
            ),
        ],
    )
    def test_build_rejects_wrong_dense_dtype(self, dataset_factory, dtype, vectors, expected_error):
        dataset = dataset_factory(num_vectors=16, dim=8)
        index = _create_dense_dispatch_index(dtype, dataset.dim)

        with pytest.raises(ValueError, match=expected_error):
            index.build(vectors(dataset), dataset.ids, dataset.num_elements, dataset.dim)

    @pytest.mark.parametrize(
        ("dtype", "query", "expected_error"),
        [
            ("float16", lambda dim: np.ones(dim, dtype=np.float32), "vector must be numpy.float16 array"),
            (
                "bfloat16",
                lambda dim: np.ones(dim, dtype=np.float16),
                r"vector must be numpy.uint16 \(raw bfloat16 bits\) array",
            ),
        ],
    )
    def test_knn_search_rejects_wrong_dense_dtype(self, dataset_factory, dtype, query, expected_error):
        dataset = dataset_factory(num_vectors=16, dim=8)
        index = _create_dense_dispatch_index(dtype, dataset.dim)

        build_vectors = dataset.base_vectors.astype(np.float16)
        if dtype == "bfloat16":
            build_vectors = _float32_to_bfloat16_bits(dataset.base_vectors)
        index.build(build_vectors, dataset.ids, dataset.num_elements, dataset.dim)

        with pytest.raises(ValueError, match=expected_error):
            index.knn_search(query(dataset.dim), 3, '{"hgraph": {"ef_search": 20}}')

    @pytest.mark.parametrize(
        ("dtype", "query", "expected_error"),
        [
            ("float16", lambda dim: np.ones(dim, dtype=np.float32), "point must be numpy.float16 array"),
            (
                "bfloat16",
                lambda dim: np.ones(dim, dtype=np.float16),
                r"point must be numpy.uint16 \(raw bfloat16 bits\) array",
            ),
        ],
    )
    def test_range_search_rejects_wrong_dense_dtype(self, dataset_factory, dtype, query, expected_error):
        dataset = dataset_factory(num_vectors=16, dim=8)
        index = _create_dense_dispatch_index(dtype, dataset.dim)

        build_vectors = dataset.base_vectors.astype(np.float16)
        if dtype == "bfloat16":
            build_vectors = _float32_to_bfloat16_bits(dataset.base_vectors)
        index.build(build_vectors, dataset.ids, dataset.num_elements, dataset.dim)

        with pytest.raises(ValueError, match=expected_error):
            index.range_search(query(dataset.dim), 1.0, '{"hgraph": {"ef_search": 20}}')


class TestRemoveVectors:
    """Tests for remove method"""

    def test_remove_single_vector(self, dataset):
        """Test removing a single vector"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        initial_count = index.get_num_elements()

        ids_to_remove = np.array([0], dtype=np.int64)
        removed = index.remove(ids_to_remove)

        assert removed == 1

    def test_remove_multiple_vectors(self, dataset):
        """Test removing multiple vectors"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        ids_to_remove = np.array([0, 1, 2], dtype=np.int64)
        removed = index.remove(ids_to_remove)

        assert removed == 3

    def test_remove_nonexistent_id(self, dataset):
        """Test removing non-existent ID"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        ids_to_remove = np.array([999999], dtype=np.int64)
        removed = index.remove(ids_to_remove)

        assert removed == 0


class TestCalDistanceById:
    """Tests for cal_distance_by_id method"""

    def test_cal_distance_by_id_single(self, dataset):
        """Test calculating distance for a single ID"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        query = dataset.query_vectors[: dataset.dim]
        ids = np.array([0, 1, 2], dtype=np.int64)

        distances = index.cal_distance_by_id(query, ids)

        assert len(distances) == 3
        assert all(d >= 0 for d in distances)

    def test_cal_distance_by_id_matches_search(self, dataset):
        """Test cal_distance_by_id returns similar distance as knn_search"""
        index_param = """
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        """
        index = create_index("hgraph", index_param)
        build_index(index, dataset)

        query = dataset.query_vectors[: dataset.dim]
        ids = np.array([0, 1, 2], dtype=np.int64)

        calc_dists = index.cal_distance_by_id(query, ids)

        assert len(calc_dists) == 3
        valid_count = sum(1 for d in calc_dists if d >= 0)
        assert valid_count > 0
