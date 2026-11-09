# Binary Vectors and Hamming Distance

VSAG can index bit-packed binary vectors, such as binary descriptors, binary embeddings or the sign
bits of float embeddings, and compare them with the Hamming distance:

```text
hamming(a, b) = popcount(a XOR b)
```

The distance is the number of bit positions where two vectors differ. Binary vectors are supported
by the [`brute_force`](../indexes/brute_force.md) (exact) and [`hgraph`](../indexes/hgraph.md)
(graph) indexes, from C++ and from `pyvsag`.

## Representation

- `dim` is the **number of bits** of a vector, not the number of bytes. It does not have to be a
  multiple of eight.
- A vector is stored in `ceil(dim / 8)` contiguous bytes, and the vectors of a dataset are packed
  back to back: a dataset of `n` vectors is a buffer of `n * ceil(dim / 8)` bytes.
- Bits are packed **MSB-first**: logical bit `i` is `(bytes[i / 8] >> (7 - i % 8)) & 1`. Bit 0 is
  the mask `0x80` of byte 0 and bit 7 is the mask `0x01` of byte 0. This is the layout produced by
  `numpy.packbits(bits, axis=-1)` with its default `bitorder="big"`, and the order in which a bit
  string such as `"10110..."` is read from left to right.
- When `dim % 8 != 0`, the low `8 - dim % 8` bits of the last byte of every vector are padding.
  Their values never change a distance: the kernels mask them, and stored codes keep them zero.

For example, with `dim = 13` every vector takes two bytes. Bits 0 to 7 are byte 0 (bit 0 is
`0x80`, bit 7 is `0x01`), bits 8 to 12 are the masks `0x80`, `0x40`, `0x20`, `0x10` and `0x08` of
byte 1, and the masks `0x04`, `0x02` and `0x01` of byte 1 are padding.

## Configuration

Binary vectors are selected with `"dtype": "binary"`, which requires `"metric_type": "hamming"`:

```json
{
    "dtype": "binary",
    "metric_type": "hamming",
    "dim": 256,
    "index_param": {
        "max_degree": 32,
        "ef_construction": 200
    }
}
```

- `index_param` does not need a quantization type: binary indexes always store the packed bits
  exactly with the `binary` quantizer. `"base_quantization_type": "binary"` (and, for HGraph,
  `"precise_quantization_type": "binary"`) may be written explicitly; any other quantization type
  is rejected because the other quantizers read numeric elements.
- `dim` must be between 1 and 16,777,216 (2^24) bits, so that every distance is an exact integer
  in the `float` result type.

These combinations are rejected with an `INVALID_ARGUMENT` error:

| Configuration | Error |
| --- | --- |
| `"metric_type": "hamming"` with `float32`, `float16`, `bfloat16`, `int8` or `sparse` | `parameters[metric_type]=hamming requires parameters[dtype]=binary` |
| `"dtype": "binary"` with `l2`, `ip` or `cosine` | `parameters[dtype]=binary requires parameters[metric_type]=hamming` |
| `"dtype": "binary"` with an index other than `brute_force` and `hgraph` | `dtype binary is supported only by brute_force and hgraph` |
| A quantization type other than `binary` | `parameters[...] must be binary when dtype is binary` |
| `"dtype": "binary"` with `"repr"` other than `dense`, or `dim` above 2^24 | parameter error naming the field |

## Supported Indexes and Operations

| Operation | `brute_force` | `hgraph` |
| --- | --- | --- |
| Build, Add | yes | yes |
| k-NN search | exact | approximate (graph search), exact distances |
| Range search (`distance <= radius`) | exact | approximate candidates, exact distances |
| Remove (`MARK_REMOVE`, `FORCE_REMOVE`) | yes | yes (`FORCE_REMOVE` needs `"support_force_remove": true` and `"use_reverse_edges": true`) |
| `CalcDistanceById` / `CalcDistancesById` with a Dataset query | yes | yes |
| `GetRawVectorByIds` | returns the packed bits | returns the packed bits |
| `UpdateVector` | yes | yes (without `force_update`, rejects a vector that moves closer to a neighbor than to its old position) |
| `GetStats` / `AnalyzeIndexBySearch` | not provided by BruteForce | yes (analysis queries carry binary vectors) |
| Serialize / Deserialize | yes | yes |
| Search with an ID `Filter` | yes | yes |

HGraph options that need floating-point input are not available for binary vectors and fail at
index creation or use with an explicit error: transform quantization (`tq_chain`, `mrle_dim`),
RaBitQ codes and `rabitq_fused_datacell`, the PiPNN builder, MCI cliques and graph enhancement.
`Tune` returns `false` because binary codes cannot be swapped for another quantizer. The overloads that take a
`const float*` query (`CalcDistanceById(const float*, ...)`) are float32-only; pass the binary
query in a `Dataset` instead.

## Distance Semantics

- The distance is the number of differing bits among the first `dim` bits. It is returned as a
  `float` whose value is always an exact integer between `0` and `dim`.
- k-NN search returns the vectors with the smallest distances first. Many vectors can share a
  distance; the order among equal distances is not specified.
- Range search returns every vector whose distance is less than or equal to the radius, so
  `radius = 0` returns exact duplicates of the query.

## Input Validation

C++ datasets carry binary vectors through `Dataset::BinaryVectors(const uint8_t*)` and
`Dataset::GetBinaryVectors()`. `Dim()` is the number of bits; the buffer must hold
`NumElements() * ceil(Dim() / 8)` bytes. A dataset without binary vectors is rejected by a binary
index, and a query whose `Dim()` differs from the index dimension is rejected. The buffer follows
the ownership rules of the other vector fields: with `Owner(true)` it is released by the dataset
allocator, or with `delete[]` when no allocator is set, and `DeepCopy` / `Append` copy
`ceil(dim / 8)` bytes per row.

`pyvsag` accepts binary vectors as C-contiguous `numpy.uint8` arrays:

| Input | Accepted shapes | Rejected with `ValueError` |
| --- | --- | --- |
| `build` / `add` vectors | `(num_elements, ceil(dim / 8))` or `(num_elements * ceil(dim / 8),)` | another dtype, another shape or length, a non-contiguous array |
| `knn_search` / `range_search` / `calc_distances_by_id` query | `(ceil(dim / 8),)` or `(1, ceil(dim / 8))` | another dtype or a wrong number of bytes |

Arrays are never converted implicitly. A non-contiguous view, for example `packed[:, ::2]`, is
rejected; pass `numpy.ascontiguousarray(packed)` to copy it.

## Example (C++)

```cpp
const int64_t dim = 100;                   // bits
const int64_t bytes = (dim + 7) / 8;       // 13 bytes per vector
std::vector<uint8_t> vectors(num_vectors * bytes);  // packed MSB-first
std::vector<int64_t> ids(num_vectors);

auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids.data())->BinaryVectors(vectors.data())->Owner(false);

auto index = vsag::Factory::CreateIndex(
    "hgraph", R"({"dtype": "binary", "metric_type": "hamming", "dim": 100,
                 "index_param": {"max_degree": 32, "ef_construction": 200}})").value();
index->Build(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->BinaryVectors(query_bytes)->Owner(false);
auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 100}})").value();
// result->GetDistances()[i] is the number of differing bits of result->GetIds()[i]
```

The complete program, which also checks every returned distance against a scalar popcount, is
[`examples/cpp/115_index_binary_hamming.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/115_index_binary_hamming.cpp).

## Example (Python)

```python
import json
import numpy as np
import pyvsag

dim = 100
bits = np.random.default_rng(0).integers(0, 2, size=(5000, dim)).astype(bool)
base = np.packbits(bits, axis=-1)            # shape (5000, 13), dtype uint8
ids = np.arange(len(base), dtype=np.int64)

index = pyvsag.Index("hgraph", json.dumps(
    {"dtype": "binary", "metric_type": "hamming", "dim": dim,
     "index_param": {"max_degree": 32, "ef_construction": 200}}))
index.build(base, ids, len(base), dim)

query = np.packbits(bits[0])                 # 13 bytes
result_ids, distances = index.knn_search(query, 10, json.dumps({"hgraph": {"ef_search": 100}}))
```

A bit string such as `"10110011..."` becomes a query with
`np.packbits(np.array([c == "1" for c in s]))`. See
[`examples/python/115_index_binary_hamming.py`](https://github.com/antgroup/vsag/blob/main/examples/python/115_index_binary_hamming.py).

## Distance Kernels

The Hamming kernels live in `src/simd/hamming_simd.{h,cpp}`. A generic scalar implementation
(64-bit words and `__builtin_popcountll`) is the reference and the fallback. Optimized versions
are selected at run time from the instructions the CPU reports:

| Implementation | Instructions | Selected when |
| --- | --- | --- |
| `avx512vpopcntdq` | `_mm512_popcnt_epi64` on 512-bit blocks, masked tail loads | AVX-512 VPOPCNTDQ |
| `avx512` | 4-bit lookup (`vpshufb`) and `vpsadbw` on 512-bit blocks | AVX-512 F/BW/DQ/VL |
| `avx2` | 4-bit lookup (`vpshufb`) and `vpsadbw` on 256-bit blocks | AVX2 |
| `sse` | scalar `POPCNT` on 64-bit words | SSE4.2 build and POPCNT reported |
| `sve` | `svcnt_u64` with `svwhilelt_b8` predicated loads | SVE |
| `neon` | `vcntq_u8` with pairwise widening accumulation | NEON |
| `generic` | portable scalar code | always |

Every implementation returns exactly the same value as the generic one. Besides one code per
call, each implementation has two batch forms: `HammingDistanceBatch4` compares a query with four
codes at arbitrary addresses, as graph search needs, and `HammingDistanceScan` computes the
distances to codes stored back to back, as `brute_force` scans need. With AVX-512 VPOPCNTDQ the
scan handles eight codes per step with a single horizontal reduction, and packs two to eight codes
into each register when a code takes 8, 16 or 32 bytes. The
[Hamming kernel benchmark](../resources/hamming_benchmark.md) measures them and reports the
recall and throughput of the binary indexes.
