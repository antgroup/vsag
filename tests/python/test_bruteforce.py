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

from test_base import TestBase
from test_dataset import TestDataset
import json
import os
import pyvsag


class TestBruteforce(TestBase):
    """Test Bruteforce index"""

    type_name: str = "brute_force"

    def __init__(
        self,
        dim: int = 128,
        num_vectors: int = 1000,
        metric: str = "l2",
        expect_recall: float = 0.9,
        quantization_type: str = "sq8",
    ):
        self.dim = dim
        self.num_vectors = num_vectors
        self.metric = metric
        self.dataset = TestDataset(dim=dim, num_vectors=num_vectors, metric=metric)
        qs = quantization_type.split(",")
        self.base_quantization_type = qs[0]
        self.precise_quantization_type = "fp32"
        if len(qs) > 1:
            self.precise_quantization_type = qs[1]
            self.use_reorder = True
        else:
            self.use_reorder = False

        self.index_params = json.dumps(
            {
                "dtype": "float32",
                "metric_type": self.metric,
                "dim": self.dim,
                "index_param": {
                    "base_quantization_type": self.base_quantization_type,
                    "precise_quantization_type": self.precise_quantization_type,
                },
            }
        )

    def init_index(self) -> pyvsag.Index:
        """Initialize Bruteforce index"""
        return TestBase.FactoryIndex(self.type_name, self.index_params)

    def general_test_search(self):
        """Test search index"""
        search_params = json.dumps({})
        TestBase.TestKnnSearch(self.index, self.dataset, search_params)

    def test_build(self):
        """Test build index"""
        self.index = self.init_index()
        TestBase.BuildIndex(self.index, self.dataset)
        self.general_test_search()

    def test_load_save(self):
        """Test load and save index"""
        filename = TestBase.GenerateRandomFilePath()
        TestBase.SaveIndex(self.index, filename)
        self.index = self.init_index()
        TestBase.LoadIndex(self.index, filename)
        os.remove(filename)
        self.general_test_search()


def run_bruteforce_test():
    """Run Bruteforce index tests"""
    metric_types = ["ip"]
    dims = [128, 256, 1024]
    quantizer_recalls = [
        ("sq8", 0.9),
        ("fp16", 0.98),
        ("fp32", 0.9999),
    ]
    for metric in metric_types:
        for dim in dims:
            for qt, recall in quantizer_recalls:
                test_brute_force = TestBruteforce(
                    dim=dim, metric=metric, expect_recall=recall, quantization_type=qt
                )
                test_brute_force.test_build()
                test_brute_force.test_load_save()


if __name__ == "__main__":
    run_bruteforce_test()
