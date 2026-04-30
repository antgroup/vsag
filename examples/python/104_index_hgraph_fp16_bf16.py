#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pyvsag
import numpy as np
import json


def float32_to_bfloat16_bits(f32_array):
    """Convert float32 numpy array to raw bfloat16 bits (uint16).

    Bfloat16 has the same exponent as float32 but only 8 bits of mantissa
    (vs 23 bits in float32). The conversion simply truncates the lower 16 bits.
    """
    f32_view = f32_array.view(np.uint32)
    bf16_bits = (f32_view >> 16).astype(np.uint16)
    return bf16_bits


def fp16_example():
    dim = 128
    num_elements = 10000
    query_elements = 1

    ids = np.arange(num_elements, dtype=np.int64)
    data_f32 = np.random.random((num_elements, dim)).astype(np.float32)
    data_fp16 = data_f32.astype(np.float16)
    data_fp16_flat = data_fp16.reshape(-1)

    query_f32 = np.random.random((query_elements, dim)).astype(np.float32)
    query_fp16 = query_f32.astype(np.float16)

    index_params = json.dumps(
        {
            "dtype": "float16",
            "metric_type": "l2",
            "dim": dim,
            "index_param": {
                "base_quantization_type": "fp16",
                "max_degree": 26,
                "ef_construction": 100,
                "alpha": 1.2,
            },
        }
    )

    print("[FP16] Create hgraph index")
    index = pyvsag.Index("hgraph", index_params)

    print("[FP16] Build hgraph index")
    index.build_fp16(
        vectors=data_fp16_flat, ids=ids, num_elements=num_elements, dim=dim
    )

    print("[FP16] KNN search hgraph index")
    search_params = json.dumps({"hgraph": {"ef_search": 100}})
    for q in query_fp16:
        result_ids, result_dists = index.knn_search_fp16(
            vector=q, k=10, parameters=search_params
        )
        print("result_ids:", result_ids)
        print("result_dists:", result_dists)

    print("[FP16] Range search hgraph index")
    for q in query_fp16:
        result_ids, result_dists = index.range_search_fp16(
            point=q, threshold=0.5, parameters=search_params
        )
        print("result_ids:", result_ids)
        print("result_dists:", result_dists)

    print("[FP16] Add vectors to hgraph index")
    add_ids = np.arange(num_elements, num_elements + 100, dtype=np.int64)
    add_data_f32 = np.random.random((100, dim)).astype(np.float32)
    add_data_fp16 = add_data_f32.astype(np.float16).reshape(-1)
    index.add_fp16(
        vectors=add_data_fp16, ids=add_ids, num_elements=100, dim=dim
    )
    print(f"[FP16] Index now has {index.get_num_elements()} elements")


def bf16_example():
    dim = 128
    num_elements = 10000
    query_elements = 1

    ids = np.arange(num_elements, dtype=np.int64)
    data_f32 = np.random.random((num_elements, dim)).astype(np.float32)
    data_bf16_flat = float32_to_bfloat16_bits(data_f32.reshape(-1))

    query_f32 = np.random.random((query_elements, dim)).astype(np.float32)
    query_bf16 = float32_to_bfloat16_bits(query_f32.reshape(-1))

    index_params = json.dumps(
        {
            "dtype": "bfloat16",
            "metric_type": "l2",
            "dim": dim,
            "index_param": {
                "base_quantization_type": "bf16",
                "max_degree": 26,
                "ef_construction": 100,
                "alpha": 1.2,
            },
        }
    )

    print("[BF16] Create hgraph index")
    index = pyvsag.Index("hgraph", index_params)

    print("[BF16] Build hgraph index")
    index.build_bf16(
        vectors=data_bf16_flat, ids=ids, num_elements=num_elements, dim=dim
    )

    print("[BF16] KNN search hgraph index")
    search_params = json.dumps({"hgraph": {"ef_search": 100}})
    for i in range(query_elements):
        q = query_bf16[i * dim : (i + 1) * dim]
        result_ids, result_dists = index.knn_search_bf16(
            vector=q, k=10, parameters=search_params
        )
        print("result_ids:", result_ids)
        print("result_dists:", result_dists)

    print("[BF16] Range search hgraph index")
    for i in range(query_elements):
        q = query_bf16[i * dim : (i + 1) * dim]
        result_ids, result_dists = index.range_search_bf16(
            point=q, threshold=0.5, parameters=search_params
        )
        print("result_ids:", result_ids)
        print("result_dists:", result_dists)

    print("[BF16] Add vectors to hgraph index")
    add_ids = np.arange(num_elements, num_elements + 100, dtype=np.int64)
    add_data_f32 = np.random.random((100, dim)).astype(np.float32)
    add_data_bf16 = float32_to_bfloat16_bits(add_data_f32.reshape(-1))
    index.add_bf16(
        vectors=add_data_bf16, ids=add_ids, num_elements=100, dim=dim
    )
    print(f"[BF16] Index now has {index.get_num_elements()} elements")


if __name__ == "__main__":
    fp16_example()
    print()
    bf16_example()
