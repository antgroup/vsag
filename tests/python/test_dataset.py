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

import numpy as np


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


if __name__ == "__main__":
    test_dataset = TestDataset(num_vectors=1000, dim=128, seed=49, metric="ip")
    print(test_dataset.gt_ids)
    print(test_dataset.gt_dists)
