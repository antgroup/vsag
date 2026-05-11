# HDF5 Dataset Format

VSAG's evaluation and benchmark tooling (most notably
[`eval_performance`](eval.md)) consumes datasets in the HDF5 format used by
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks). This page
documents the exact layout VSAG expects so you can prepare custom datasets or
debug failing evaluations.

The dataset layout described below is the **dense** layout (selected by the global
attribute `type="dense"`, or by omitting the attribute). For **sparse** datasets
(`type="sparse"`), `/train` and `/test` are 1-D `INT8` byte streams produced by VSAG's
sparse-vector serialization (decoded by `parse_sparse_vectors` in
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

## References

- Public benchmark datasets compatible with this layout are available from
  [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  (e.g. `sift-128-euclidean.hdf5`, `gist-960-euclidean.hdf5`).
- See [Evaluation Tool](eval.md) for how datasets in this format are consumed.
