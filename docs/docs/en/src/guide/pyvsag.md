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
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

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

The current `save` and `load` wrappers do not propagate errors returned by the underlying
serialization and deserialization operations. Verify that the output file was created and
round-trip important indexes before relying on persisted data.

## Index Operations

The Python binding exposes the following index-management methods in addition to build, search,
save, and load:

| Method | Result | Notes |
| --- | --- | --- |
| `get_num_elements()` | `int` | Current number of elements. |
| `get_memory_usage()` | `int` | Index memory usage in bytes. |
| `check_id_exist(id)` | `bool` | Whether an ID currently exists. |
| `get_min_max_id()` | `(min_id, max_id)` | Returns `(-1, -1)` if the operation fails. |
| `add(vectors, ids, num_elements, dim)` | `None` | Adds a contiguous dense vector matrix; dtype must match the index. |
| `remove(ids)` | `int` | Removes a one-dimensional `int64` ID array. Returns `0` both when no ID is removed and when the underlying operation fails. |
| `cal_distance_by_id(query, ids)` | `numpy.ndarray` | Returns one float32 distance per ID. Invalid IDs produce `-1`; an underlying operation failure leaves the entire result at `-1`. |

```python
# Add two dense vectors.
new_vectors = np.random.random((2, dim)).astype(np.float32)
new_ids = np.array([10_001, 10_002], dtype=np.int64)
index.add(new_vectors, new_ids, len(new_ids), dim)

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
array containing BF16 bit patterns. Operation support remains index-dependent. Input validation
and operations such as `add` raise Python exceptions, but `remove`, `cal_distance_by_id`, `save`,
and `load` use the sentinel or unchecked behaviors described above instead of consistently
propagating underlying operation failures.

## Relationship with the C++ Library

`pyvsag` wraps the same `vsag::Index` API as C++ and shares the underlying index binaries. You can
build an index in Python and load it in C++ (and vice versa) as long as parameters match.

## More Examples

See `examples/python/` in the repository.
