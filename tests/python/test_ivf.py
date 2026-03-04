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

"""Pytest tests for IVF index"""

import json
import pyvsag
import pytest
from conftest import create_index, build_index, save_index, load_index, verify_knn_search


TYPE_NAME = "ivf"


def _create_index_params(dim: int, metric: str, quantization_type: str) -> str:
    """Create index parameters for IVF"""
    qs = quantization_type.split(",")
    base_quantization_type = qs[0]
    precise_quantization_type = "fp32"
    use_reorder = False
    if len(qs) > 1:
        precise_quantization_type = qs[1]
        use_reorder = True
    
    return json.dumps(
        {
            "dtype": "float32",
            "metric_type": metric,
            "dim": dim,
            "index_param": {
                "buckets_count": 50,
                "partition_strategy_type": "ivf",
                "ivf_train_type": "kmeans",
                "base_quantization_type": base_quantization_type,
                "precise_quantization_type": precise_quantization_type,
                "use_reorder": use_reorder,
            },
        }
    )


@pytest.mark.parametrize("metric", ["ip"])
@pytest.mark.parametrize("dim", [128, 256, 1024])
@pytest.mark.parametrize("quantization_type,expect_recall", [
    ("sq8", 0.9),
    ("fp16", 0.92),
    ("fp32", 0.93),
    ("sq8,fp16", 0.92),
    ("sq8_uniform,fp32", 0.9),
])
def test_ivf_build(dim, metric, quantization_type, expect_recall, dataset_factory):
    """Test building IVF index"""
    dataset = dataset_factory(dim=dim, num_vectors=1000, metric=metric)
    index_params = _create_index_params(dim, metric, quantization_type)
    index = create_index(TYPE_NAME, index_params)
    build_index(index, dataset)
    
    search_params = json.dumps({"ivf": {"scan_buckets_count": 50}})
    verify_knn_search(index, dataset, search_params, expect_recall=expect_recall)


@pytest.mark.parametrize("metric", ["ip"])
@pytest.mark.parametrize("dim", [128, 256, 1024])
@pytest.mark.parametrize("quantization_type,expect_recall", [
    ("sq8", 0.9),
    ("fp16", 0.92),
    ("fp32", 0.93),
    ("sq8,fp16", 0.92),
    ("sq8_uniform,fp32", 0.9),
])
def test_ivf_load_save(dim, metric, quantization_type, expect_recall, dataset_factory, tmp_path):
    """Test loading and saving IVF index"""
    dataset = dataset_factory(dim=dim, num_vectors=1000, metric=metric)
    index_params = _create_index_params(dim, metric, quantization_type)
    
    # Build and save
    index = create_index(TYPE_NAME, index_params)
    build_index(index, dataset)
    filename = tmp_path / f"test_ivf_{dim}_{metric}_{quantization_type.replace(',', '_')}.vsag"
    save_index(index, str(filename))
    
    # Load and test
    index = create_index(TYPE_NAME, index_params)
    load_index(index, str(filename))
    search_params = json.dumps({"ivf": {"scan_buckets_count": 50}})
    verify_knn_search(index, dataset, search_params, expect_recall=expect_recall)
