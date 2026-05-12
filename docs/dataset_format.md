# HDF5 Dataset Structure Documentation

This document covers two on-disk variants used by the `eval_performance` tool:

- **Dense vectors** (default; backwards-compatible with public datasets).
- **Sparse vectors** (selected via the file-level `type` attribute).

The variant is identified by the file-level `type` attribute (see
[Vector type](#vector-type) below). When the attribute is absent, the
file is interpreted as **dense** for compatibility with existing public
benchmark datasets.

---

## Vector type

### `type` (file-level attribute)
- **Type**: String (ASCII), optional
- **Values**:
    - `"dense"` (or attribute missing): `/train` and `/test` are
      `(N, D)` matrices, see [Dense layout](#dense-layout).
    - `"sparse"`: `/train` and `/test` are one-dimensional byte streams,
      see [Sparse layout](#sparse-layout).

---

## Dense layout

### Mandatory Datasets

### `/train` (Training Data)
- **Type**: `INT8` or `FLOAT32`
- **Dimensions**: `(N, D)`
    - `N`: Number of base vectors (`number_of_base`)
    - `D`: Feature dimensionality (`dim`)
- **Description**: Contains feature vectors for index construction. Data type is inferred from HDF5:
    - `H5T_INTEGER` (1-byte) → `INT8`
    - `H5T_FLOAT` (4-byte) → `FLOAT32`

### `/test` (Query Data)
- **Type**: Must match `/train` type (`INT8` or `FLOAT32`)
- **Dimensions**: `(Q, D)`
    - `Q`: Number of query vectors (`number_of_query`)
    - `D`: Same dimensionality as `/train`
- **Validation**: Column count must equal `/train`'s `D`

### `/neighbors` (True Neighbor Indices)
- **Type**: `INT64`
- **Dimensions**: `(Q, K)`
    - `Q`: Matches `/test` row count
    - `K`: Number of neighbors per query
- **Content**: Precomputed ground truth indices from training set

### `/distances` (True Distance Values)
- **Type**: `FLOAT32`
- **Dimensions**: `(Q, K)` (identical to `/neighbors`)
- **Note**: Must align with neighbor indices

---

## Global Attributes

### `distance` (Metric Definition)
- **Type**: String (ASCII)
- **Required**: Yes
- **Values**:
    - `"euclidean"`: Computed as `sqrt(L2Sqr)`
    - `"ip"`: Inner product (auto-detects data type)
    - `"angular"`: Normalized inner product similarity

---

## Optional Datasets

### `/train_labels` & `/test_labels`
- **Type**: `INT64`
- **Dimensions**:
    - `/train_labels`: `(N,)`
    - `/test_labels`: `(Q,)`
- **Requirement**: Both must exist if labels are present

### `/valid_ratios`
- **Type**: `FLOAT32`
- **Dimensions**: `(L,)` where `L` = number of unique labels
- **Usage**: Stores per-class validation ratios

---

## Structural Requirements

1. **Dimensional Compatibility**:
    - `train_shape[1] == test_shape[1]` (same `D`)
    - `neighbors.shape == distances.shape`

2. **Type Mapping**:
   | HDF5 Specification       | Internal Type | Size  | Used In               |
   |--------------------------|---------------|-------|-----------------------|
   | `H5T_INTEGER` (size=1)   | `INT8`        | 1 byte| `/train`, `/test`     |
   | `H5T_FLOAT` (size=4)     | `FLOAT32`     | 4 bytes| `/train`, `/test`    |
   | `H5T_INTEGER` (size=8)   | `INT64`       | 8 bytes| Label datasets       |

3. **Memory Organization**:
    - Row-major storage for all matrices
    - Feature vectors stored contiguously:
        - `/train` size = `N × D × data_size` (1 or 4 bytes/element)

---

## Sparse layout

When the file-level `type` attribute equals `"sparse"`, the `/train` and
`/test` datasets do **not** follow the `(N, D)` dense matrix layout.
Instead they are stored as a one-dimensional `INT8` (`H5T_INTEGER` of
size 1) dataset whose payload is a raw byte stream of packed sparse
vectors. The `int8` storage class is a transport detail only; the bytes
are not int8 vector elements.

### `/train`, `/test` (sparse byte stream)
- **HDF5 type**: `H5T_INTEGER`, size 1 (`INT8`)
- **HDF5 shape**: 1-D, total length = sum of per-vector record sizes
- **Endianness**: little-endian
- **Content**: a contiguous sequence of records, one per sparse vector,
  in order. Each record has the layout:

  | Field      | Type        | Size            | Description                              |
  |------------|-------------|-----------------|------------------------------------------|
  | `len`      | `uint32`    | 4 bytes         | Number of non-zero entries in the vector |
  | `ids[len]` | `uint32[]`  | `4 * len` bytes | Feature indices (column ids)             |
  | `vals[len]`| `float32[]` | `4 * len` bytes | Values associated with `ids`             |

  Records are concatenated back-to-back with no padding or separators.
  A `len == 0` record is allowed and occupies only the 4-byte length
  field.

- **Key ordering**: on load, the eval tool sorts each vector's `ids` in
  ascending order (and reorders `vals` accordingly). Writers may emit
  unordered keys, but readers should not depend on that.

### Distance metric

- **`distance` attribute**: only `"ip"` (inner product) is supported by
  the C++ eval tool for sparse datasets. The recorded ground-truth
  distance is `1 - <q1, q2>`.

### Ground truth

`/neighbors` and `/distances` follow the same shape/type rules as in the
[Dense layout](#dense-layout):

- `/neighbors`: `INT64`, shape `(Q, K)`
- `/distances`: `FLOAT32`, shape `(Q, K)`

### Python helper

The Python package `pyvsag` ships a decoder in `pyvsag.sparse`:

```python
from pyvsag.sparse import load_sparse_hdf5

data = load_sparse_hdf5("sparse.hdf5")
# data["type"]     -> "sparse"
# data["distance"] -> "ip"
# data["train"]    -> list[dict[int, float]]   one dict per sparse vector
# data["test"]     -> list[dict[int, float]]
# data["neighbors"], data["distances"] -> numpy arrays
```

Use `pyvsag.sparse.decode_sparse_bytes(buffer)` directly if you already
have the raw bytes from another source.

### Reference implementation

The byte-stream encoder/decoder lives in C++ at
`tools/eval/eval_dataset.cpp` (see `parse_sparse_vectors` and
`serialize_sparse_vectors`).
