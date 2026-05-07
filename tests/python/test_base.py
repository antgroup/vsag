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

"""Test base module - re-exported from conftest for backward compatibility"""

import pyvsag
import numpy as np
import os
from conftest import (
    TestDataset,
    create_index,
    build_index,
    save_index,
    load_index,
    calculate_recall,
)

__all__ = ["TestBase", "TestDataset"]


class TestBase:
    """Test base class for VSAG index tests - backward compatibility wrapper"""

    @staticmethod
    def FactoryIndex(index_type: str, index_param: str) -> pyvsag.Index:
        """Factory method to create index for testing"""
        return create_index(index_type, index_param)

    @staticmethod
    def BuildIndex(index: pyvsag.Index, dataset: TestDataset):
        """Build index for testing"""
        build_index(index, dataset)

    @staticmethod
    def SaveIndex(index: pyvsag.Index, file_path: str):
        """Serialize index for testing"""
        save_index(index, file_path)

    @staticmethod
    def LoadIndex(index: pyvsag.Index, file_path: str):
        """Load index for testing"""
        load_index(index, file_path)

    @staticmethod
    def CalRecall(results: np.ndarray, gt_ids: np.ndarray, topk: int = 10) -> float:
        """Calculate recall for testing"""
        return calculate_recall(results, gt_ids, topk)

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
        from conftest import verify_knn_search
        verify_knn_search(index, dataset, search_param, topk, expect_recall)
