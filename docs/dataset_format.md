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
    - `"sparse"`: `/train` and `/test` are flat byte streams of shape
      `(X,)`, see [Sparse layout](#sparse-layout).

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
Instead they are stored as a flat `INT8` (`H5T_INTEGER` of size 1)
dataset whose payload is a raw byte stream of packed sparse vectors.
Calling `f["/train"].shape` from h5py returns `(X,)` where `X` is the
total number of bytes; the `int8` storage class is a transport detail
only — the bytes are not int8 vector elements.

### `/train`, `/test` (sparse byte stream)
- **HDF5 type**: `H5T_INTEGER`, size 1 (`INT8`)
- **HDF5 shape**: `(X,)`, where `X` is the total byte-stream length
  (sum of all per-vector record sizes)
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

### `/train_offsets`, `/test_offsets` (random-access index, optional)

These two datasets store the per-record byte offsets into the
corresponding `/train` and `/test` byte streams so that the i-th sparse
vector can be located in **O(1)** without scanning the stream.

- **HDF5 type**: `H5T_INTEGER`, size 8 (`UINT64`)
- **HDF5 shape**: `(N + 1,)` for `/train_offsets` and `(Q + 1,)` for
  `/test_offsets`
- **Content**: `offsets[i]` is the byte offset where record `i` starts
  inside the matching byte stream; `offsets[N]` is the sentinel and
  equals the total byte stream size, so the size of record `i` is
  `offsets[i + 1] - offsets[i]`. The array is non-decreasing.

Both datasets are **optional**. Writers in this repository always emit
them when writing sparse files, but legacy sparse HDF5 files that only
contain `/train` and `/test` keep loading: the offsets are recomputed on
load by walking the byte stream once. When the on-disk offsets are
present, they are cross-checked against the recomputed offsets and the
file is rejected as corrupted on any mismatch.

Example random access (Python):

```python
import h5py, numpy as np

with h5py.File("sparse.hdf5") as f:
    buf = f["/train"][:]                        # shape (X,), INT8 byte stream
    off = f["/train_offsets"][:]                # shape (N+1,), UINT64

def get_sparse_record(byte_stream, offsets, i):
    start, end = int(offsets[i]), int(offsets[i + 1])
    rec = byte_stream[start:end].tobytes()
    ln = int(np.frombuffer(rec, dtype="<u4", count=1)[0])
    ids = np.frombuffer(rec, dtype="<u4", count=ln, offset=4)
    vals = np.frombuffer(rec, dtype="<f4", count=ln, offset=4 + ln * 4)
    return ids, vals
```

### `/train_token_sequences`, `/test_token_sequences` (optional)

These two datasets carry the **original tokenized document** that
produced each sparse vector. They are entirely optional: sparse HDF5
files that omit both datasets still load correctly. When present, they
must appear in lockstep with `/train` and `/test`: the i-th record in
`/train_token_sequences` corresponds to the i-th sparse vector in
`/train` (same for `/test`).

- **HDF5 type**: `H5T_INTEGER`, size 1 (`INT8`)
- **HDF5 shape**: `(X,)`, where `X` is the total byte-stream length
  (sum of all per-record sizes)
- **Endianness**: little-endian
- **Content**: a contiguous sequence of records, one per sparse vector,
  in the same order as `/train` / `/test`. Each record has the layout:

  | Field            | Type        | Size                | Description                                  |
  |------------------|-------------|---------------------|----------------------------------------------|
  | `seq_len`        | `uint32`    | 4 bytes             | Number of tokens in the original document    |
  | `term_ids[seq_len]` | `uint32[]` | `4 * seq_len` bytes | Term ids in tokenization order (duplicates and order are preserved) |

  Records are concatenated back-to-back with no padding or separators.
  A `seq_len == 0` record is allowed and occupies only the 4-byte
  length field; readers should treat it as "no original document
  available for this vector".

- **Number of records**: must equal the number of sparse vectors in the
  matching split. Readers raise an error if the counts disagree or if
  the stream is truncated.
- **Ordering vs. `ids`**: `term_ids` are stored in the original token
  order (with possible duplicates). This is intentionally **different**
  from `ids`, which the loader sorts ascending and deduplicates.

### `/train_token_sequences_offsets`, `/test_token_sequences_offsets` (required when sequences are present)

Whenever `/train_token_sequences` (resp. `/test_token_sequences`) is
present, the paired `UINT64` offset index **must** also be present.

- **HDF5 type**: `H5T_INTEGER`, size 8 (`UINT64`)
- **HDF5 shape**: `(N + 1,)` (resp. `(Q + 1,)`)
- **Content**: identical contract to `/train_offsets` —
  `offsets[i]` is the byte offset of the i-th token-sequence record,
  and `offsets[N]` equals the total byte stream length. This makes
  per-record random access O(1) without scanning the stream.

Contract: the byte-stream dataset and its offsets dataset **live or die
together**. Readers reject the file if exactly one of the pair exists
(either a `*_token_sequences` dataset without its `*_offsets`, or vice
versa). When both are present, the on-disk offsets are cross-checked
against the offsets rebuilt from the byte stream; a mismatch is treated
as corruption and aborts the load.

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
