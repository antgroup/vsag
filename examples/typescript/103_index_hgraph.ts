// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { Index } from "vsag";

function hgraphTest(): void {
    const numVectors = 1000;
    const dim = 128;
    const ids = new BigInt64Array(numVectors);
    const vectors = new Float32Array(dim * numVectors);

    for (let i = 0; i < numVectors; i++) {
        ids[i] = BigInt(i);
    }
    for (let i = 0; i < dim * numVectors; i++) {
        vectors[i] = Math.random();
    }

    const indexParams = JSON.stringify({
        dtype: "float32",
        metric_type: "l2",
        dim,
        index_param: {
            base_quantization_type: "fp32",
            max_degree: 16,
            ef_construction: 100,
            alpha: 1.2,
        },
    });

    const index = new Index("hgraph", indexParams);
    index.build(vectors, ids, numVectors, dim);

    const queryVector = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        queryVector[i] = Math.random();
    }

    const searchParams = JSON.stringify({ hgraph: { ef_search: 100 } });
    const topk = 10;
    const { ids: resultIds, distances } = index.knnSearch(queryVector, topk, searchParams);

    for (let i = 0; i < topk; i++) {
        console.log(`${resultIds[i]}: ${distances[i]}`);
    }

    const filename = "/tmp/103_index_hgraph_ts.index";
    index.save(filename);

    const newIndex = new Index("hgraph", indexParams);
    newIndex.load(filename);
    console.log(`After load, index contains: ${newIndex.getNumElements()} elements`);
}

hgraphTest();
