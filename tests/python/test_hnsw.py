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

"""Pytest tests for HNSW index"""

import json
import pyvsag
import pytest
from conftest import create_index, build_index, save_index, load_index, verify_knn_search


TYPE_NAME = "hnsw"


def _create_index_params(dim: int, metric: str) -> str:
    """Create index parameters for HNSW"""
    max_degree = int(dim / 4)
    ef_construction = max(200, max_degree)
    return json.dumps(
        {
            "dtype": "float32",
            "metric_type": metric,
            "dim": dim,
            "hnsw": {
                "max_degree": max_degree,
                "ef_construction": ef_construction,
            },
        }
    )


@pytest.mark.parametrize("metric", ["ip"])
@pytest.mark.parametrize("dim", [128, 256, 1024])
def test_hnsw_build(dim, metric, dataset_factory):
    """Test building HNSW index"""
    dataset = dataset_factory(dim=dim, num_vectors=1000, metric=metric)
    index_params = _create_index_params(dim, metric)
    index = create_index(TYPE_NAME, index_params)
    build_index(index, dataset)
    
    search_params = json.dumps({"hnsw": {"ef_search": 40}})
    verify_knn_search(index, dataset, search_params)


@pytest.mark.parametrize("metric", ["ip"])
@pytest.mark.parametrize("dim", [128, 256, 1024])
def test_hnsw_load_save(dim, metric, dataset_factory, tmp_path):
    """Test loading and saving HNSW index"""
    dataset = dataset_factory(dim=dim, num_vectors=1000, metric=metric)
    index_params = _create_index_params(dim, metric)
    
    # Build and save
    index = create_index(TYPE_NAME, index_params)
    build_index(index, dataset)
    filename = tmp_path / f"test_hnsw_{dim}_{metric}.vsag"
    save_index(index, str(filename))
    
    # Load and test
    index = create_index(TYPE_NAME, index_params)
    load_index(index, str(filename))
    search_params = json.dumps({"hnsw": {"ef_search": 40}})
    verify_knn_search(index, dataset, search_params)
