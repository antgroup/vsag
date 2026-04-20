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

```python
import numpy as np
import pyvsag

dim = 128
n = 10_000
data = np.random.random((n, dim)).astype(np.float32)
ids = np.arange(n, dtype=np.int64)

index = pyvsag.Index(
    "hgraph",
    dim=dim,
    metric_type="l2",
    dtype="float32",
    index_param={"base_quantization_type": "fp32", "max_degree": 32, "ef_construction": 300},
)

index.build(ids=ids, vectors=data)

query = np.random.random((1, dim)).astype(np.float32)
ids_out, dists_out = index.knn_search(query, k=10, search_param={"hgraph": {"ef_search": 60}})
print(ids_out, dists_out)
```

## Serialization

```python
index.serialize("index.bin")

new_index = pyvsag.Index("hgraph", dim=dim, metric_type="l2", dtype="float32",
                        index_param={...})
new_index.deserialize("index.bin")
```

## Relationship with the C++ Library

`pyvsag` wraps the same `vsag::Index` API as C++ and shares the underlying index binaries. You can
build an index in Python and load it in C++ (and vice versa) as long as parameters match.

## More Examples

See `examples/python/` in the repository.
