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

"""
SINDI SQ8 (quantized posting-list) + DMQ8 reorder example.

This example demonstrates how to build and search a SINDI index with:
  - SQ8 quantization enabled for posting-list storage (use_quantization: true)
  - DMQ8 reorder enabled for high-precision distance computation (rerank_type: "dmq8")

The SQ8 posting-list quantization reduces memory footprint during the inverted-list phase,
while DMQ8 reranking provides fast approximate distance computation with quantized codes.
"""

import numpy as np
import json
import pyvsag


def cal_recall(index, index_pointers, indices, values, ids, k, search_params):
    """Calculate recall@k for the index."""
    correct = 0
    res_ids, res_dists = index.knn_search(
        index_pointers, indices, values, k, search_params
    )
    for i in range(len(ids)):
        if ids[i] == res_ids[i][0]:
            correct += 1

    return correct / len(ids)


def convert_to_csr(vectors_with_metadata):
    """
    Convert a list of metadata-augmented sparse vectors to CSR format.

    Args:
        vectors_with_metadata (list of dict): Each item has:
            {
                "id": int,              # Business-level vector ID (e.g., item/user ID)
                "features": dict        # Sparse features: {feature_dim_index: value}
            }

    Returns:
        tuple:
            - index_pointers (np.ndarray, uint32): CSR index_pointers array, shape (batch_size + 1,)
            - indices (np.ndarray, uint32): Feature column indices, shape (nnz,)
            - values (np.ndarray, float32): Non-zero values, shape (nnz,)
            - ids (np.ndarray, int64): Original IDs for result mapping, shape (batch_size,)
    """
    index_pointers = [0]
    indices = []
    values = []
    ids = []

    for item in vectors_with_metadata:
        vid = item["id"]
        features = item["features"]

        ids.append(vid)

        sorted_features = sorted(features.items())

        for feat_idx, feat_val in sorted_features:
            indices.append(int(feat_idx))
            values.append(float(feat_val))

        index_pointers.append(len(indices))

    return (
        np.array(index_pointers, dtype=np.uint32),
        np.array(indices, dtype=np.uint32),
        np.array(values, dtype=np.float32),
        np.array(ids, dtype=np.int64),
    )


def sindi_sq8_dmq8_reorder_test():
    """
    Build SINDI index with SQ8 posting-list quantization and DMQ8 reorder.

    The configuration enables:
      - "use_quantization": true  -> SQ8 quantization on posting values
      - "use_reorder": true       -> Enable reranking phase
      - "rerank_type": "dmq8"     -> Use the 8-bit DMQ backend for reranking
    """
    # Sparse vectors in DICT format.
    vectors_in_dict = [
        {"id": 1001, "features": {0: 1.0, 3: 2.0}},
        {"id": 1002, "features": {1: 1.5, 2: 1.0, 4: 3.0}},
        {"id": 1003, "features": {0: 0.8, 1: 0.9, 2: 1.1}},
        {"id": 1004, "features": {0: 1.2, 2: 0.5, 5: 2.5}},
        {"id": 1005, "features": {1: 2.0, 3: 1.5, 4: 1.0}},
    ]

    index_pointers, indices, values, ids = convert_to_csr(vectors_in_dict)

    # Build index with SQ8 + DMQ8 reorder configuration
    index_params = json.dumps(
        {
            "dtype": "sparse",
            "dim": 128,
            "metric_type": "ip",
            "index_param": {
                "use_quantization": True,  # Enable SQ8 quantization on posting values
                "use_reorder": True,       # Enable reranking phase
                "rerank_type": "dmq8",     # Use the 8-bit DMQ backend
                "doc_prune_ratio": 0.0,
                "window_size": 60000,
            },
        }
    )

    index = pyvsag.Index("sindi", index_params)

    print("[build] Creating SINDI index with SQ8 + DMQ8 reorder configuration...")
    index.build(index_pointers=index_pointers, indices=indices, values=values, ids=ids)
    print("[build] Index created successfully")

    # Search with reorder parameters
    search_params = json.dumps(
        {
            "sindi": {
                "query_prune_ratio": 0.2,  # Prune 20% of least-relevant query terms
                "term_prune_ratio": 0.1,   # Prune 10% of least-relevant terms from posting lists
                "n_candidate": 10,         # Keep top 10 candidates after inverted phase for reranking
            }
        }
    )

    recall = cal_recall(index, index_pointers, indices, values, ids, 1, search_params)
    print(f"[search] SINDI SQ8 + DMQ8 recall@1: {recall:.4f}")

    # Save and reload
    filename = "./python_example_sindi_sq8_dmq8.index"
    print(f"[save] Saving index to {filename}...")
    index.save(filename)

    # Deserialize and test again
    print("[load] Loading index from disk...")
    index = pyvsag.Index("sindi", index_params)
    index.load(filename)

    recall = cal_recall(index, index_pointers, indices, values, ids, 1, search_params)
    print(f"[deserialize] SINDI SQ8 + DMQ8 recall@1: {recall:.4f}")

    print("\nConfiguration summary:")
    print("  - use_quantization: true  (SQ8 posting-list storage)")
    print("  - use_reorder: true        (Enable reranking)")
    print("  - rerank_type: dmq8        (8-bit DMQ backend)")
    print("\nBenefits:")
    print("  - Reduced memory footprint via SQ8 quantization")
    print("  - Fast reranking with DMQ8 quantized codes")
    print("  - Trade-off between memory and accuracy vs FP32 reranking")


if __name__ == "__main__":
    sindi_sq8_dmq8_reorder_test()
