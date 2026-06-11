# HDF5 Dataset Format

VSAG's evaluation and benchmark tooling (most notably
[`eval_performance`](eval.md)) consumes datasets in the HDF5 format used by
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks). This page
documents the exact layout VSAG expects so you can prepare custom datasets or
debug failing evaluations.

The dataset layout described below is the **dense** layout (selected by the global
attribute `type="dense"`, or by omitting the attribute). For **sparse** datasets
(`type="sparse"`), `/train` and `/test` are flat `INT8` byte streams of shape `(X,)`
produced by VSAG's sparse-vector serialization (decoded by `parse_sparse_vectors` in
`tools/eval/eval_dataset.cpp`); all other datasets and attributes below still apply.

## Mandatory Datasets

### `/train` (base vectors)

- **Type**: `INT8` or `FLOAT32`
- **Shape**: `(N, D)`
    - `N` — number of base vectors (`number_of_base`)
    - `D` — feature dimensionality (`dim`)
- **Notes**: the element type is inferred from HDF5:
    - `H5T_INTEGER` (1-byte) → `INT8`
    - `H5T_FLOAT` (4-byte) → `FLOAT32`

### `/test` (query vectors)

- **Type**: must match `/train`
- **Shape**: `(Q, D)`
    - `Q` — number of query vectors (`number_of_query`)
    - `D` — must equal `/train`'s `D`

### `/neighbors` (ground-truth indices)

- **Type**: `INT64`
- **Shape**: `(Q, K)`
    - `K` — number of ground-truth neighbors per query
- **Content**: precomputed top-`K` indices into `/train`.

### `/distances` (ground-truth distances)

- **Type**: `FLOAT32`
- **Shape**: `(Q, K)` (identical to `/neighbors`)
- **Note**: each entry must align with the same position in `/neighbors`.

## Global Attributes

### `type` (vector type)

- **Type**: ASCII string
- **Required**: no (defaults to `"dense"` if the attribute is missing)
- **Allowed values**:
    - `"dense"` — dense vectors stored as standard matrices in `/train` and `/test`
    - `"sparse"` — sparse vectors stored in the serialized format produced by VSAG's
      sparse-vector helpers

### `distance` (metric definition)

The evaluation tool treats `distance` values as **distances** (smaller is better) when
comparing against the ground truth in `/distances`. Prepare ground-truth distances using the
formulas below.

- **Type**: ASCII string
- **Required**: yes
- **Allowed values for dense vectors**:
    - `"euclidean"` — L2 distance, computed as `sqrt(L2Sqr)`
    - `"ip"` — inner-product distance (`1 - inner_product`); data type auto-detected
    - `"angular"` — cosine distance (`1 - cosine_similarity`)
- **Allowed values for sparse vectors**:
    - `"ip"` — sparse inner-product distance (`1 - sparse_inner_product`); other metrics
      are not supported for sparse vectors
- **Allowed values for multi-vector**:
    - Same as dense vectors (`"euclidean"`, `"ip"`, `"angular"`); multi-vector uses the
      same per-sub-vector distance function as dense vectors

## Optional Datasets

### `/train_labels` and `/test_labels`

- **Type**: `INT64`
- **Shapes**:
    - `/train_labels`: `(N,)`
    - `/test_labels`: `(Q,)`
- **Requirement**: if labels are present, both datasets must exist.

### `/valid_ratios`

- **Type**: `FLOAT32`
- **Shape**: `(L,)`
- **Usage**: stores per-class validation ratios. The evaluation tool indexes this array
  with the **raw label value** (`valid_ratio_[label]`, see
  `tools/eval/eval_dataset.h:71`), so labels must be non-negative integers and `L` must
  be strictly greater than the maximum label value (typically `L > max(label)` with valid
  indices `0..L-1`). It is the dataset author's responsibility to keep the array large
  enough to cover every label that appears in `/train_labels` and `/test_labels`.

## Multi-Vector Datasets

When `type="multi_vector"`, the file uses a flat-expanded layout where each document’s
sub-vectors are concatenated into a single 2D matrix, and a companion `vector_counts`
array records how many sub-vectors belong to each document.

### Additional Global Attribute

| Attribute           | Type    | Required | Description                                          |
|---------------------|---------|----------|------------------------------------------------------|
| `multi_vector_dim`  | `INT64` | yes      | Sub-vector dimensionality (number of floats per sub-vector) |

### Additional Datasets

| Dataset                  | Shape                    | Type     | Description                                                        |
|--------------------------|--------------------------|----------|--------------------------------------------------------------------|
| `/train_multi_vectors`   | `(sum_counts_train, D)`  | `FLOAT32`| All training sub-vectors, flat-concatenated row by row              |
| `/test_multi_vectors`    | `(sum_counts_test, D)`   | `FLOAT32`| All query sub-vectors, flat-concatenated row by row                 |
| `/train_vector_counts`   | `(N,)`                   | `UINT32` | Number of sub-vectors per training document                         |
| `/test_vector_counts`    | `(Q,)`                   | `UINT32` | Number of sub-vectors per query document                            |

> `D` equals `multi_vector_dim`. `sum_counts_train` is the sum of all values in
> `/train_vector_counts`, and `sum_counts_test` is the sum of all values in
> `/test_vector_counts`.

When `type="multi_vector"`, the standard `/train` and `/test` datasets are **not
required** — the document count (`N`, `Q`) is derived from `/train_vector_counts`
and `/test_vector_counts` instead. All other datasets (`/neighbors`, `/distances`,
optional labels) remain mandatory.

The evaluation tool reconstructs one `vsag::MultiVector` per document from the
flat array plus the counts, then passes the full array to
`vsag::Dataset::MultiVectors()`, `VectorCounts()`, and `MultiVectorDim()`.

## Structural Requirements

1. **Dimensional compatibility**
    - `train_shape[1] == test_shape[1]` (same `D`)
    - `neighbors.shape == distances.shape`
2. **Type mapping**

   | HDF5 Specification     | Internal Type | Size    | Used In                                          |
   |------------------------|---------------|---------|--------------------------------------------------|
   | `H5T_INTEGER` (size=1) | `INT8`        | 1 byte  | `/train`, `/test`                                |
   | `H5T_FLOAT` (size=4)   | `FLOAT32`     | 4 bytes | `/train`, `/test`, `/distances`, `/valid_ratios` |
   | `H5T_INTEGER` (size=8) | `INT64`       | 8 bytes | `/neighbors`, `/train_labels`, `/test_labels`    |

3. **Memory organization**
    - Row-major storage for all matrices.
    - Feature vectors stored contiguously:
        - `/train` total size = `N × D × element_size` (1 or 4 bytes per element).

## Sparse layout

When the global attribute `type` equals `"sparse"`, `/train` and `/test` do **not** follow
the `(N, D)` dense matrix layout. They are instead stored as flat `INT8`
(`H5T_INTEGER`, size 1) datasets whose payload is a raw byte stream of packed sparse
vectors. Calling `f["/train"].shape` from h5py returns `(X,)` where `X` is the total
number of bytes; the `int8` storage class is a transport detail only — the bytes are
not int8 vector elements.

### `/train`, `/test` (sparse byte stream)

- **HDF5 type**: `H5T_INTEGER`, size 1 (`INT8`)
- **HDF5 shape**: `(X,)`, where `X` is the total byte-stream length
  (sum of all per-vector record sizes)
- **Endianness**: little-endian
- **Content**: a contiguous sequence of records, one per sparse vector, in order. Each
  record has the following fields, concatenated with no padding or separators:

  | Field        | Type        | Size            | Description                              |
  |--------------|-------------|-----------------|------------------------------------------|
  | `len`        | `uint32`    | 4 bytes         | Number of non-zero entries in the vector |
  | `ids[len]`   | `uint32[]`  | `4 * len` bytes | Feature indices (column ids)             |
  | `vals[len]`  | `float32[]` | `4 * len` bytes | Values associated with `ids`             |

  A `len == 0` record is allowed and occupies only the 4-byte length field.

- **Key ordering**: on load, the eval tool sorts each vector's `ids` in ascending order
  (and reorders `vals` accordingly). Writers may emit unordered keys, but readers should
  not rely on that.

### `/train_offsets`, `/test_offsets` (random-access index, optional)

These two datasets store the per-record byte offsets into the matching
`/train` and `/test` sparse byte streams so that the i-th sparse vector
can be located in **O(1)** without scanning the stream.

- **HDF5 type**: `H5T_INTEGER`, size 8 (`UINT64`)
- **HDF5 shape**: `(N + 1,)` for `/train_offsets` and `(Q + 1,)` for
  `/test_offsets`
- **Content**: `offsets[i]` is the byte offset of record `i`;
  `offsets[N]` is the sentinel and equals the total byte stream length.
  The size of record `i` is `offsets[i + 1] - offsets[i]`. The array is
  non-decreasing.

Both datasets are **optional**. VSAG writers always emit them when
writing sparse files, but legacy sparse files that only contain `/train`
and `/test` keep loading: the offsets are recomputed on load by walking
the byte stream once. When the on-disk offsets are present, they are
cross-checked against the recomputed offsets and the file is rejected as
corrupted on any mismatch.

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

  | Field               | Type        | Size                | Description                                  |
  |---------------------|-------------|---------------------|----------------------------------------------|
  | `seq_len`           | `uint32`    | 4 bytes             | Number of tokens in the original document    |
  | `term_ids[seq_len]` | `uint32[]`  | `4 * seq_len` bytes | Term ids in tokenization order (duplicates and order are preserved) |

  Records are concatenated with no padding or separators. A
  `seq_len == 0` record is allowed and occupies only the 4-byte length
  field; readers should treat it as "no original document available
  for this vector".

- **Number of records**: must equal the number of sparse vectors in the
  matching split. Readers raise an error if counts disagree or if the
  stream is truncated.
- **Ordering vs. `ids`**: `term_ids` are stored in the original token
  order (duplicates kept). This is intentionally **different** from
  `ids`, which the loader sorts ascending.

### `/train_token_sequences_offsets`, `/test_token_sequences_offsets` (required when sequences are present)

Whenever `/train_token_sequences` (resp. `/test_token_sequences`) is
present, the paired `UINT64` offset index **must** also be present.

- **HDF5 type**: `H5T_INTEGER`, size 8 (`UINT64`)
- **HDF5 shape**: `(N + 1,)` (resp. `(Q + 1,)`)
- **Content**: same contract as `/train_offsets`, enabling O(1) random
  access to the i-th token-sequence record.

Contract: the byte-stream dataset and its offsets dataset **live or die
together**. Readers reject the file if exactly one of the pair exists
(either a `*_token_sequences` dataset without its `*_offsets`, or vice
versa). When both are present, the on-disk offsets are cross-checked
against the offsets rebuilt from the byte stream; a mismatch is treated
as corruption and aborts the load.

### Ground truth and metric

`/neighbors` and `/distances` follow the same shape and type rules as in the dense
layout above. Only `"ip"` (sparse inner-product distance, `1 - sparse_inner_product`)
is supported via the `distance` attribute.

### Python helper

The Python package `pyvsag` ships a decoder in [`pyvsag.sparse`](https://github.com/antgroup/vsag/blob/main/python/pyvsag/sparse.py):

```python
from pyvsag.sparse import load_sparse_hdf5

data = load_sparse_hdf5("sparse.hdf5")
# data["type"]      -> "sparse"
# data["distance"]  -> "ip"
# data["train"]     -> list[dict[int, float]]   one dict per sparse vector, keys ascending
# data["test"]      -> list[dict[int, float]]
# data["neighbors"] -> numpy.ndarray  shape (Q, K) int64
# data["distances"] -> numpy.ndarray  shape (Q, K) float32
```

`pyvsag.sparse.decode_sparse_bytes(buffer)` is also exposed for callers that already
hold the raw byte stream.

### Reference implementation

The byte-stream encoder/decoder lives at
[`tools/eval/eval_dataset.cpp`](https://github.com/antgroup/vsag/blob/main/tools/eval/eval_dataset.cpp)
(see `parse_sparse_vectors` and `serialize_sparse_vectors`).

## References

- Public benchmark datasets compatible with this layout are available from
  [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  (e.g. `sift-128-euclidean.hdf5`, `gist-960-euclidean.hdf5`).
- See [Evaluation Tool](eval.md) for how datasets in this format are consumed.
