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

function hnswTest(): void {
    /******************* Prepare Base Dataset *****************/
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

    /******************* Create HNSW Index *****************/
    const indexParams = JSON.stringify({
        dtype: "float32",
        metric_type: "l2",
        dim: dim,
        hnsw: {
            max_degree: 16,
            ef_construction: 100,
        },
    });

    console.log("[Create] hnsw index");
    const index = new Index("hnsw", indexParams);

    /******************* Build HNSW Index *****************/
    console.log("[Build] hnsw index");
    index.build(vectors, ids, numVectors, dim);
    console.log(`After build, index contains: ${index.getNumElements()} elements`);

    /******************* KnnSearch For HNSW Index *****************/
    const queryVector = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        queryVector[i] = Math.random();
    }

    const searchParams = JSON.stringify({
        hnsw: {
            ef_search: 100,
        },
    });

    console.log("[Search] hnsw index");
    const topk = 10;
    const { ids: resultIds, distances } = index.knnSearch(queryVector, topk, searchParams);

    /******************* Print Search Result *****************/
    console.log("results:");
    for (let i = 0; i < topk; i++) {
        console.log(`  ${resultIds[i]}: ${distances[i]}`);
    }

    /******************* Save and Load *****************/
    const filename = "/tmp/101_index_hnsw_ts.index";
    console.log(`[Save] index to ${filename}`);
    index.save(filename);

    const newIndex = new Index("hnsw", indexParams);
    console.log(`[Load] index from ${filename}`);
    newIndex.load(filename);
    console.log(`After load, index contains: ${newIndex.getNumElements()} elements`);
}

hnswTest();
