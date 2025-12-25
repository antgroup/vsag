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

from test_dataset import TestDataset
import pyvsag
import json
import numpy as np
import os


class TestBase:
    """Test base class for VSAG index tests"""

    @staticmethod
    def FactoryIndex(index_type: str, index_param: str) -> pyvsag.Index:
        """Factory method to create index for testing"""
        return pyvsag.Index(index_type, index_param)

    @staticmethod
    def BuildIndex(index: pyvsag.Index, dataset: TestDataset):
        """Build index for testing"""
        index.build(
            dataset.base_vectors, dataset.ids, dataset.num_elements, dataset.dim
        )

    @staticmethod
    def SaveIndex(index: pyvsag.Index, file_path: str):
        """Serialize index for testing"""
        index.save(file_path)

    @staticmethod
    def LoadIndex(index: pyvsag.Index, file_path: str):
        """Load index for testing"""
        index.load(file_path)

    @staticmethod
    def CalRecall(results: np.ndarray, gt_ids: np.ndarray, topk: int = 10) -> float:
        """Calculate recall for testing"""
        # too many indices for array: array is 1-dimensional, but 2 were indexed
        recall = np.isin(results[:topk], gt_ids[:topk]).sum(axis=0)
        return recall / topk

    @staticmethod
    def GenerateRandomFilePath() -> str:
        """Generate empty file path for testing"""
        random_id = np.random.randint(0, 1000000)
        filename = f"test_index_{random_id}.vsag"
        while os.path.exists(filename):
            random_id = np.random.randint(0, 1000000)
            filename = f"test_index_{random_id}.vsag"
        return filename

    @staticmethod
    def TestKnnSearch(
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
            recall += TestBase.CalRecall(ids, dataset.gt_ids, topk)
        print(f"recall: {recall / query_count}")
        assert recall >= expect_recall * query_count


if __name__ == "__main__":
    dataset = TestDataset()
    index_params = json.dumps(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "hnsw": {"max_degree": 16, "ef_construction": 100},
        }
    )
    index = TestBase.FactoryIndex("hnsw", index_params)
    TestBase.BuildIndex(index, dataset)
    TestBase.TestKnnSearch(index, dataset, json.dumps({"hnsw": {"ef_search": 100}}))
