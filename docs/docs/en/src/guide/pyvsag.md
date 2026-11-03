# pyvsag

`pyvsag` is the official Python binding for VSAG, implemented with pybind11. Sources live under
`python_bindings/` and `python/`.

## Installation

```bash
pip install pyvsag
```

To build from source:

```bash
make pyvsag PY_VERSION=3.10
# Build wheels for multiple Python versions:
make pyvsag-all
```

## Quick Start

`pyvsag.Index(name, parameters)` accepts the index name and a JSON-encoded parameter string,
matching the C++ `vsag::Factory::CreateIndex` signature:

```python
import json
import numpy as np
import pyvsag

dim = 128
num_elements = 10_000

data = np.random.random((num_elements, dim)).astype(np.float32)
ids = np.arange(num_elements, dtype=np.int64)

index_params = json.dumps({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": dim,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 300,
    },
})

index = pyvsag.Index("hgraph", index_params)
failed_ids = index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)
if failed_ids:
    print("IDs not inserted:", failed_ids)

query = np.random.random(dim).astype(np.float32)
search_params = json.dumps({"hgraph": {"ef_search": 60}})
result_ids, result_dists = index.knn_search(
    vector=query, k=10, parameters=search_params,
)
print(result_ids, result_dists)
```

## Saving & Loading

```python
index.save("index.bin")

new_index = pyvsag.Index("hgraph", index_params)
new_index.load("index.bin")
```

`save` and `load` retain the ordinary C++ binary stream format. Core serialization/deserialization
failures raise `pyvsag.VsagError`; file-open and final stream-state failures raise `OSError`.
These checks do not make saving transactional: an existing file can be truncated before an error,
and a failed load does not promise rollback. The binding adds no guarantee that deleted-ID state
is preserved across a native round trip; the native persistence issue remains pending. Round-trip
important indexes and verify the behavior required by your application.

## Index Operations

The Python binding exposes the following index-management methods in addition to build, search,
save, and load:

| Method | Result | Notes |
| --- | --- | --- |
| `get_num_elements()` | `int` | Current number of elements. |
| `get_memory_usage()` | `int` | Index memory usage in bytes. |
| `check_id_exist(id)` | `bool` | Whether an ID currently exists. |
| `get_min_max_id()` | `(min_id, max_id)` | Core failures raise `VsagError`; support and empty-index behavior depend on the backend. |
| `add(vectors, ids, num_elements, dim)` | `list[int]` | IDs that failed insertion; overall failures raise `VsagError`. Dense dtype must match the index. |
| `remove(ids)` | `int` | Number actually removed; `0` means a successful no-op. Core failures raise `VsagError`. |
| `calc_distances_by_id(query, ids)` / `cal_distance_by_id(query, ids)` (legacy alias) | `numpy.ndarray` | Returns one float32 distance per ID in a 1D array. Missing IDs produce `-1`; underlying invalid/unsupported operations raise `VsagError` with the C++ error code and message. |

```python
# Add two dense vectors.
new_vectors = np.random.random((2, dim)).astype(np.float32)
new_ids = np.array([10_001, 10_002], dtype=np.int64)
failed_ids = index.add(new_vectors, new_ids, len(new_ids), dim)

assert index.check_id_exist(10_001)
print(index.get_num_elements(), index.get_memory_usage())
print(index.get_min_max_id())

# Calculate distances from one query to a candidate list.
candidate_ids = np.array([10_001, 10_002], dtype=np.int64)
distances = index.cal_distance_by_id(query, candidate_ids)

# Remove IDs; the return value is the number actually removed.
removed = index.remove(candidate_ids)
```

`add` accepts a flat contiguous array or a row-major two-dimensional matrix. For
`dtype: "float16"`, pass `numpy.float16`; for `dtype: "bfloat16"`, pass a `numpy.uint16`
array containing BF16 bit patterns. Operation support remains index-dependent.

### Return values, errors, and migration

| API / input | Contract |
| --- | --- |
| `build` (dense and CSR overloads), `add` | Return `list[int]` of IDs rejected by the native backend, including partial failures. `[]` means no rejected IDs were reported, not a backend-independent deduplication guarantee. For example, duplicate handling differs between IVF and HGraph/BruteForce. Overall failures raise rather than becoming a failed-ID list. |
| Dense `knn_search`, `range_search` and their `_with_statistics` variants | Return 1D ID and distance arrays of the actual result count, which may be less than `k` or zero. No padding or sentinel is added. Statistics variants append the native JSON statistics string. |
| CSR overloads of `knn_search` and `knn_search_with_statistics` | Keep the existing rectangular `(num_queries, k)` contract. Each row has a valid-result prefix followed by ID `-1` and distance `+inf` padding. The statistics variant appends one JSON string per query row. |
| Core operation failure | `pyvsag.VsagError`, a `RuntimeError` subclass, preserves the native message, integer `.code` (`vsag::ErrorType`), and string `.operation`. This applies to construction, ordinary/statistics searches, build/add, removal, min/max, distance calculation, and persistence. |
| File-open / final stream failure | `OSError` from `save` or `load`; native serialization errors still use `VsagError`. |
| Binding validation failure | `ValueError` for invalid shape, dense dtype/layout, CSR consistency, or size overflow. Python argument/type conversion can also raise `TypeError`. |

Migration: callers that previously expected `None` from build/add should inspect the failed-ID list.
Replace fixed-`k` dense loops with `len(result_ids)` or `zip(result_ids, result_dists)`. Do not treat
an empty result, `remove == 0`, or `(-1, -1)` as a substitute for catching an operation failure.
Existing `except RuntimeError` handlers still catch `VsagError`; use its attributes for structured
handling instead of parsing the message.

For sparse batches, identify padding from the distance, **not from the ID alone**: `-1` can be a
legitimate native ID. For finite native distances, a convenient per-row mask is:

```python
valid = ~np.isposinf(result_dists[row])
row_ids = result_ids[row][valid]
row_dists = result_dists[row][valid]
```

The padding contract does not normalize native non-finite distance values. Empty indexes, empty
queries/batches, `k == 0`, unsupported operations, and duplicate IDs retain public C++/backend
semantics; there is no new blanket rejection or cross-backend normalization. In particular, native
early-empty returns can precede backend parameter validation.

### Dense and CSR input layout

Dense build/add accept a flat C-contiguous array of `num_elements * dim` values or a C-contiguous
matrix of exactly `(num_elements, dim)`; single dense search queries must be 1D and C-contiguous.
These paths require the exact configured dtype (`float32`, `float16`, or raw-bit `uint16` for BF16)
and do not silently cast or copy strided dense inputs. An empty `(0, dim)` build/add matrix is a
zero-row batch, not a single query. Shape products, byte sizes, and int64/Python-size narrowing are
checked before native pointer use.

IDs and CSR arrays instead retain convertible-input behavior: they are converted to contiguous
`int64` IDs, `uint32` row pointers/indices, and `float32` values, preserving the logical order of
positive- or negative-stride views. CSR pointers must start at zero, be nondecreasing, and end at
the common indices/values length; row counts and array sizes must fit the supported integer ranges.
`index_pointers=[0]` with empty indices/values represents zero rows, whereas `[0, 0]` represents one
empty sparse row. Result arrays use explicit NumPy byte strides, including empty outputs.

### Distance binding input safety

`calc_distances_by_id` and its legacy alias accept one 1D dense query whose length must equal the index dimension, plus a 1D ID array. Short, long, and empty queries are rejected by the C++ Dataset validation, not treated as missing IDs. NumPy inputs are converted to contiguous float32 / int64 arrays as needed, so positive- and negative-stride query and ID views retain their logical order. This does not add sparse, multi-vector, or multi-query support to the binding.

The Node.js equivalents `calcDistancesById(query, ids)` and legacy `calDistanceById(query, ids)` retain their 1D `Float32Array` result. They require exactly `Float32Array` queries and `BigInt64Array` IDs (other array kinds throw `TypeError`), validate the actual query length through the C++ Dataset API, and throw `Error` with the C++ message on operation failure. Missing IDs still produce `-1`; they are not operation failures.

## Relationship with the C++ Library

`pyvsag` wraps the same `vsag::Index` API as C++ and shares the underlying index binaries. You can
build an index in Python and load it in C++ (and vice versa) as long as parameters match.

## More Examples

See `examples/python/` in the repository.
