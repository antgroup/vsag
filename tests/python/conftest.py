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

"""Pytest fixtures and utilities for VSAG index tests"""

import json
import numpy as np
import pyvsag
import pytest
import os
from pathlib import Path


class TestDataset:
    """Test dataset class for VSAG index tests"""

    def __init__(self, num_vectors=200, dim=128, seed=42, topk=10, metric="l2"):
        self.num_vectors = num_vectors
        self.dim = dim
        self.seed = seed
        self.query_vectors_count = 10
        self.base_vectors, self.query_vectors, self.ids, self.num_elements, self.dim = (
            TestDataset.create_test_dataset(num_vectors, dim, seed)
        )
        self.gt_ids, self.gt_dists = self.cal_gt_knn_results(
            self.base_vectors, self.query_vectors, k=topk, metric=metric
        )

    @staticmethod
    def generate_random_vectors(num_vectors, dim, seed=49):
        """Generate random float32 vectors for testing"""
        np.random.seed(seed)
        return np.random.randn(num_vectors * dim).astype(np.float32)

    @staticmethod
    def generate_sequential_ids(num_vectors):
        """Generate sequential IDs for testing"""
        return np.arange(num_vectors, dtype=np.int64)

    @staticmethod
    def create_test_dataset(num_vectors=100, dim=128, seed=49):
        """Create a standard test dataset"""
        vectors = TestDataset.generate_random_vectors(num_vectors, dim, seed)
        ids = TestDataset.generate_sequential_ids(num_vectors)
        query_vectors = TestDataset.generate_random_vectors(10, dim, seed * 2)
        return vectors, query_vectors, ids, num_vectors, dim

    def cal_gt_knn_results(
        self, vectors: np.ndarray, query_vectors: np.ndarray, k=10, metric="l2"
    ):
        """Calculate ground truth KNN results for verification"""
        new_vectors = vectors.reshape(-1, self.dim)
        new_query_vectors = query_vectors.reshape(-1, self.dim)
        result_ids = []
        result_dists = []
        for query_vector in new_query_vectors:
            if metric == "l2":
                dists = np.linalg.norm(new_vectors - query_vector, axis=1)
            elif metric == "ip":
                dists = -np.dot(new_vectors, query_vector)
            elif metric == "cosine":
                dists = 1 - (
                    np.dot(new_vectors, query_vector)
                    / (
                        np.linalg.norm(new_vectors, axis=1)
                        * np.linalg.norm(query_vector)
                    )
                )
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            sorted_indices = np.argsort(dists)
            result_ids.append(sorted_indices[:k])
            result_dists.append(dists[sorted_indices[:k]])
        return np.array(result_ids), np.array(result_dists)


# Fixtures

@pytest.fixture
def dataset():
    """Default test dataset (128 dim, 200 vectors, l2 metric)"""
    return TestDataset(num_vectors=200, dim=128, seed=42, topk=10, metric="l2")


@pytest.fixture
def dataset_factory():
    """Factory fixture for creating datasets with custom parameters"""
    def _create(num_vectors=200, dim=128, seed=42, topk=10, metric="l2"):
        return TestDataset(
            num_vectors=num_vectors, dim=dim, seed=seed, topk=topk, metric=metric
        )
    return _create


@pytest.fixture
def generate_random_filepath():
    """Generate random file path for testing"""
    def _generate():
        random_id = np.random.randint(0, 1000000)
        filename = f"test_index_{random_id}.vsag"
        while os.path.exists(filename):
            random_id = np.random.randint(0, 1000000)
            filename = f"test_index_{random_id}.vsag"
        return filename
    return _generate


# Helper functions

def create_index(index_type: str, index_param: str) -> pyvsag.Index:
    """Factory function to create index for testing"""
    return pyvsag.Index(index_type, index_param)


def build_index(index: pyvsag.Index, dataset: TestDataset):
    """Build index for testing"""
    index.build(
        dataset.base_vectors, dataset.ids, dataset.num_elements, dataset.dim
    )


def save_index(index: pyvsag.Index, file_path: str):
    """Serialize index for testing"""
    index.save(file_path)


def load_index(index: pyvsag.Index, file_path: str):
    """Load index for testing"""
    index.load(file_path)


def calculate_recall(results: np.ndarray, gt_ids: np.ndarray, topk: int = 10) -> float:
    """Calculate recall for testing"""
    recall = np.isin(results[:topk], gt_ids[:topk]).sum(axis=0)
    return recall / topk


def verify_knn_search(
    index: pyvsag.Index,
    dataset: TestDataset,
    search_param: str,
    topk: int = 10,
    expect_recall: float = 0.9,
):
    """Test knn_search method for testing"""
    query_count = dataset.query_vectors_count
    query_vectors = dataset.query_vectors.reshape(query_count, dataset.dim)
    recall = 0
    for i in range(query_count):
        query = query_vectors[i]
        ids, _ = index.knn_search(vector=query, k=topk, parameters=search_param)
        assert len(ids) == topk
        recall += calculate_recall(ids, dataset.gt_ids, topk)
    print(f"recall: {recall / query_count}")
    assert recall >= expect_recall * query_count
