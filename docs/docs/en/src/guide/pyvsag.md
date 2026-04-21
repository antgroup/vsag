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

## Relationship with the C++ Library

`pyvsag` wraps the same `vsag::Index` API as C++ and shares the underlying index binaries. You can
build an index in Python and load it in C++ (and vice versa) as long as parameters match.

## More Examples

See `examples/python/` in the repository.
