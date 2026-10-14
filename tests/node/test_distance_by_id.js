// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Run directly with Node >=16; no npm test runner or TypeScript build needed.
// VSAG_NODE_ADDON must name the freshly rebuilt native addon when not using
// the package's default build/Release location.
const assert = require("node:assert/strict");
const path = require("node:path");
const addon = require(path.resolve(process.env.VSAG_NODE_ADDON ||
    path.join(__dirname, "../../typescript/build/Release/vsag_node.node")));

const index = new addon.Index("hgraph", JSON.stringify({
    dtype: "float32", metric_type: "l2", dim: 4,
    index_param: {base_quantization_type: "fp32", max_degree: 16, ef_construction: 50},
}));
index.build(new Float32Array([1, 2, 3, 4, 4, 3, 2, 1]), new BigInt64Array([10n, 20n]), 2, 4);

for (const name of ["calcDistancesById", "calDistanceById"]) {
    const distance = (...args) => index[name](...args);
    const query = new Float32Array([1, 2, 3, 4]);
    const ids = new BigInt64Array([10n, 999n, 20n, 10n]);
    const actual = distance(query, ids);
    assert(actual instanceof Float32Array);
    assert.deepEqual(Array.from(actual), [0, -1, 20, 0]);
    const empty = distance(query, new BigInt64Array());
    assert(empty instanceof Float32Array);
    assert.equal(empty.length, 0);

    for (const length of [0, 3, 5]) {
        for (const candidate of [10n, 999n]) {
            assert.throws(() => distance(new Float32Array(length), new BigInt64Array([candidate])),
                error => error instanceof Error && !(error instanceof TypeError) &&
                    /calcDistancesById failed:/.test(error.message) &&
                    /dimension|dim|query|vector/i.test(error.message));
        }
    }
    for (const wrongQuery of [new Float64Array(4), new Int32Array(4), new Uint8Array(16),
        new BigInt64Array(4), [], null, new DataView(new ArrayBuffer(16))]) {
        assert.throws(() => distance(wrongQuery, ids), TypeError);
    }
    for (const wrongIds of [new BigUint64Array(1), new Int32Array(1), new Float32Array(1),
        [], null, new DataView(new ArrayBuffer(8))]) {
        assert.throws(() => distance(query, wrongIds), TypeError);
    }
    assert.throws(() => distance(), TypeError);
    assert.throws(() => distance(query), TypeError);

    // Nonzero typed-array byte offsets must still address the selected elements.
    const offsetQuery = new Float32Array([99, 1, 2, 3, 4, 99]).subarray(1, 5);
    const offsetIds = new BigInt64Array([888n, 10n, 999n, 20n, 10n, 888n]).subarray(1, 5);
    assert.deepEqual(Array.from(distance(offsetQuery, offsetIds)), [0, -1, 20, 0]);

    // This dense-only API cannot supply SINDI's required sparse representation.
    const sparse = new addon.Index("sindi", JSON.stringify({
        dim: 16, dtype: "sparse", metric_type: "ip",
        index_param: {use_reorder: true, doc_prune_ratio: 0.0,
            window_size: 10000, term_id_limit: 16},
    }));
    assert.throws(() => sparse[name](new Float32Array(16), new BigInt64Array([999n])),
        error => error instanceof Error && /calcDistancesById failed:/.test(error.message) &&
            /sparse|representation|support/i.test(error.message));
}
console.log("distance binding safety checks passed");
