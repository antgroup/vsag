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

def SINDI_test():
    # 3 elements with CSR format
    index_pointers = np.array([0, 2, 5, 8], dtype=np.int32)
    indices = np.array([0, 3, 1, 2, 4, 0, 1, 2])
    values = np.array([1.0, 2.0, 1.5, 1.0, 3.0, 0.8, 0.9, 1.1], dtype=np.float32)
    ids = np.array([1001, 1002, 1003], dtype=np.int64)

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
    SINDI_test()

