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

def hgraph_test():
    dim = 128
    num_elements = 10000
    query_elements = 1

    # Generating sample data
    ids = range(num_elements)
    data = np.float32(np.random.random((num_elements, dim)))
    query = np.float32(np.random.random((query_elements, dim)))

    # Declaring index
    index_params = """
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha":1.2
        }
    }
    """

    print("[Create] hgraph index")
    index = pyvsag.Index("hgraph", index_params)

    print("[Build] hgraph index")
    index.build(vectors=data,
                ids=ids,
                num_elements=num_elements,
                dim=dim)
    
    print("[Search] hgraph index")
    search_params = json.dumps({"hgraph": {"ef_search": 100}})
    for q in query:
        result_ids, result_dists = index.knn_search(vector=q, k=10, parameters=search_params)
        print("result_ids:", result_ids)
        print("result_dists:", result_dists)


if __name__ == '__main__':
    hgraph_test()
