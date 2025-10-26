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

import numpy as np
import json
import sys
import pyvsag

def cal_recall(index, index_pointers, indices, values, ids, k, search_params):
    correct = 0
    res_ids, res_dists = index.knn_search(index_pointers, indices, values, k, search_params)
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
        np.array(ids, dtype=np.int64)
    )

def sindi_test():
    # Sparse vectors in DICT format.
    vectors_in_dict = [
        {"id": 1001, "features": {0: 1.0, 3: 2.0}},
        {"id": 1002, "features": {1: 1.5, 2: 1.0, 4: 3.0}},
        {"id": 1003, "features": {0: 0.8, 1: 0.9, 2: 1.1}}
    ]

    index_pointers, indices, values, ids = convert_to_csr(vectors_in_dict)

    # Sparse vectors in CSR (Compressed Sparse Row) format.
    # This format is used to represent multiple sparse vectors efficiently.

    # index_pointers: array of start/end positions in `indices` and `values` for each vector.
    #   Shape: (batch_size + 1,)
    #   Example: [0, 2, 5, 8] means:
    #       Vector 0: uses values[0:2]   -> 2 non-zero elements
    #       Vector 1: uses values[2:5]   -> 3 non-zero elements
    #       Vector 2: uses values[5:8]   -> 3 non-zero elements
    assert list(index_pointers) == [0, 2, 5, 8]

    # indices: column indices (feature dimensions) of non-zero values.
    #   These are the "internal feature IDs" (e.g., vocabulary indices).
    #   Must be uint32 to match C++ interface requirements.
    assert list(indices) == [0, 3, 1, 2, 4, 0, 1, 2]

    # values: actual floating-point values of non-zero elements.
    #   Corresponds to TF-IDF weights, counts, or other sparse features.
    assert np.allclose(values, [1.0, 2.0, 1.5, 1.0, 3.0, 0.8, 0.9, 1.1])

    # ids: user-defined identifiers (labels) for each vector.
    #   Not used in computation; only for tracking results (e.g., mapping ANN results back to business entities).
    #   Example: item IDs, user IDs, document IDs.
    assert list(ids) == [1001, 1002, 1003]

    # build index
    index_params = json.dumps({
        "dtype": "sparse",
        "dim": 128,
        "metric_type": "ip",
        "index_param": {
            "doc_prune_ratio": 0.0,
            "window_size": 100000
        }
    })
    index = pyvsag.Index("sindi", index_params)

    index.build(index_pointers=index_pointers,
                indices=indices,
                values=values,
                ids=ids)

    search_params = json.dumps({
        "sindi": {
            "query_prune_ratio": 0,
            "n_candidate": 3
        }
    })

    # cal recall
    print("[build] sindi recall:", cal_recall(index, index_pointers, indices, values, ids, 1, search_params))
    filename = "./python_example_sindi.index"
    index.save(filename)

    # deserialize and cal recall
    index = pyvsag.Index("sindi", index_params)
    index.load(filename)
    print("[deserialize] sindi recall:", cal_recall(index, index_pointers, indices, values, ids, 1, search_params))


if __name__ == '__main__':
    sindi_test()

