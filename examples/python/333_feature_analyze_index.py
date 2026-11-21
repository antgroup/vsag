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

import json
import os
import struct
import tempfile

import numpy as np
import pyvsag


def analyze_index_example():
    dim = 64
    num_elements = 1000
    data = np.float32(np.random.random((num_elements, dim)))

    print("[Build] hgraph index")
    index = pyvsag.Index(
        "hgraph",
        json.dumps(
            {
                "dtype": "float32",
                "metric_type": "l2",
                "dim": dim,
                "index_param": {
                    "base_quantization_type": "sq8",
                    "max_degree": 16,
                    "ef_construction": 100,
                },
            }
        ),
    )
    index.build(vectors=data, ids=range(num_elements), num_elements=num_elements, dim=dim)

    directory = tempfile.mkdtemp()
    index_path = os.path.join(directory, "hgraph.index")
    index.save(index_path)

    # Query files hold uint32 rows, uint32 dim and the float32 vectors.
    query_path = os.path.join(directory, "queries.bin")
    with open(query_path, "wb") as stream:
        stream.write(struct.pack("<II", 10, dim))
        stream.write(data[:10].astype("<f4").tobytes())

    print("[Analyze] static analysis needs only the index file")
    report = pyvsag.analyze_index(index_path)
    print("index type:", report["metadata"]["index_type"])
    print("vectors:", report["metadata"]["num_elements"])
    print("graph components:", report["static_analysis"]["connect_components"])

    print("[Analyze] dynamic analysis with a query file")
    report = pyvsag.analyze_index(
        index_path,
        query_path=query_path,
        search_parameter={"hgraph": {"ef_search": 100}},
        topk=10,
    )
    print("recall:", report["dynamic_analysis"]["result"]["recall_query"])

    try:
        pyvsag.analyze_index(os.path.join(directory, "missing.index"))
    except pyvsag.IndexAnalysisError as error:
        print("expected error:", error.code)

    print("[Command] the same report from the shell:")
    print("pyvsag-analyze-index --index_path {} --query_path {} --topk 10".format(
        index_path, query_path))


if __name__ == "__main__":
    analyze_index_example()
