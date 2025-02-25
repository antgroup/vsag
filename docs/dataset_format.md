# HDF5 Dataset Structure Documentation

## Mandatory Datasets

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
