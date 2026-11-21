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

"""Binary vectors with the Hamming distance.

A binary vector of ``dim`` bits is passed as ``ceil(dim / 8)`` bytes of ``numpy.uint8``, packed
MSB-first exactly like ``numpy.packbits`` does by default. ``dim`` counts bits and does not need
to be a multiple of eight. The distance is the number of differing bits.
"""

import json
import sys

import numpy as np
import pyvsag

DIM = 100  # 100 bits -> 13 bytes per vector
NUM_VECTORS = 5000


def hamming(packed_a, packed_b, dim):
    """Reference distance: unpack the bits and count the differences."""
    a = np.unpackbits(packed_a, axis=-1)[..., :dim]
    b = np.unpackbits(packed_b, axis=-1)[..., :dim]
    return np.count_nonzero(a != b, axis=-1)


def main():
    rng = np.random.default_rng(47)

    # Pack boolean matrices with numpy.packbits: shape (rows, ceil(dim / 8)), dtype uint8.
    bits = rng.integers(0, 2, size=(NUM_VECTORS, DIM), dtype=np.uint8).astype(bool)
    # Bit strings work too: '1011...' -> booleans -> packed bytes.
    first = ("1011001110001111" * 7)[:DIM]
    bits[0] = np.array([c == "1" for c in first])
    base = np.packbits(bits, axis=-1)
    ids = np.arange(NUM_VECTORS, dtype=np.int64)
    print(f"base: shape={base.shape} dtype={base.dtype} ({DIM} bits per vector)")

    # The query is vector 0 with three flipped bits, so vector 0 is at distance 3.
    query_bits = bits[0].copy()
    query_bits[[2, 50, 99]] ^= True
    query = np.packbits(query_bits)

    ok = True
    build = {"dtype": "binary", "metric_type": "hamming", "dim": DIM}

    # Exact search with brute_force.
    brute_force = pyvsag.Index("brute_force", json.dumps(build))
    brute_force.build(vectors=base, ids=ids, num_elements=NUM_VECTORS, dim=DIM)
    result_ids, result_dists = brute_force.knn_search(vector=query, k=5, parameters="{}")
    expected = hamming(base[result_ids], query, DIM)
    ok &= bool(np.array_equal(result_dists, expected.astype(np.float32)))
    ok &= bool(result_ids[0] == 0 and result_dists[0] == 3)
    print("brute_force:", list(zip(result_ids.tolist(), result_dists.astype(int).tolist())))

    # Range search returns every vector within the given Hamming distance.
    range_ids, range_dists = brute_force.range_search(point=query, threshold=3.0, parameters="{}")
    all_dists = hamming(base, query, DIM)
    ok &= set(range_ids.tolist()) == set(np.nonzero(all_dists <= 3)[0].tolist())
    print("range (<= 3 bits):", range_ids.tolist())

    # Graph search with hgraph; binary codes are selected automatically.
    hgraph_params = dict(build, index_param={"max_degree": 32, "ef_construction": 200})
    hgraph = pyvsag.Index("hgraph", json.dumps(hgraph_params))
    hgraph.build(vectors=base, ids=ids, num_elements=NUM_VECTORS, dim=DIM)
    search = json.dumps({"hgraph": {"ef_search": 100}})
    result_ids, result_dists = hgraph.knn_search(vector=query, k=5, parameters=search)
    expected = hamming(base[result_ids], query, DIM)
    ok &= bool(np.array_equal(result_dists, expected.astype(np.float32)))
    print("hgraph:", list(zip(result_ids.tolist(), result_dists.astype(int).tolist())))

    distances = hgraph.calc_distances_by_id(query, np.array([0, 1, 2], dtype=np.int64))
    ok &= bool(np.array_equal(distances, all_dists[:3].astype(np.float32)))
    print("distances to ids 0..2:", distances.astype(int).tolist())

    print("all Hamming distances verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
