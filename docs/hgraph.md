# HGraph Index

## Definition
HGraph (Hierarchical Graph) is a **graph-based** index structure that constructs multiple layers of proximity graphs to achieve efficient approximate nearest neighbor search. It combines the advantages of hierarchical navigable graphs with quantization techniques for memory efficiency.

## Working Principle
1. **Graph Construction Phase**:
   First, build a hierarchical graph structure where each layer represents a different level of granularity. The bottom layer contains all data points, while upper layers contain fewer points as navigation aids. Each node maintains connections to its nearest neighbors within a maximum degree constraint.

2. **Quantization Integration**:
   Optionally apply quantization techniques (SQ8, PQ, FP16, etc.) to compress vector data, significantly reducing memory usage while maintaining search accuracy. Supports both base quantization and reordering with higher precision.

3. **Search Phase**:
   When a query vector is given, start from the top layer of the hierarchical graph and perform a greedy search to find the nearest neighbor. Then use this as the entry point for the next layer, progressively refining the search until reaching the bottom layer for the final result.

## Suitable Scenarios
1. High-dimensional vector scenarios, typically 128-4096 dimensions.
2. High-accuracy requirements for nearest neighbor search.
3. Memory-constrained scenarios where quantization can significantly reduce storage.
4. Dynamic datasets requiring support for incremental updates and deletions.
5. Real-time search scenarios requiring low-latency responses.

## Usage
For examples, refer to [103_index_hgraph.cpp](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp).

For RabitQ split 1bit + 7bit storage/search, see [rabitq_split_1bit_7bit.md](rabitq_split_1bit_7bit.md).

## Factory Parameter Overview Table

| **Category** | **Parameter** | **Type** | **Default Value** | **Required** | **Description** |
|--------------|---------------|----------|-------------|--------------|-----------------|
| **Basic** | dtype | string | "float32" | Yes | Data type (only float32 supported) |
| **Basic** | metric_type | string | "l2" | Yes | Distance metric: l2, ip, cosine |
| **Basic** | dim | int | - | Yes | Vector dimension [1, 4096] |
| **Quantization** | base_quantization_type | string | - | Yes | Base quantization type |
| **Quantization** | base_codes_type | string | "flatten" | No | Base code storage type; use "rabitq_split" for RabitQ split 1bit + 7bit storage |
| **Quantization** | rabitq_version | string | "standard" | No | RabitQ implementation version; use "split_1bit_7bit" with RabitQ split storage |
| **Quantization** | rabitq_error_rate | float | 1.9 | No | Error-rate multiplier used by RabitQ one-bit lower-bound search |
| **Quantization** | use_reorder | bool | false | No | Enable result reordering/reranking after base search |
| **Quantization** | reorder_source | string | "precise" | Conditional | Source used for reorder when `use_reorder=true`: "precise" uses precise codes, "base" uses base codes |
| **Quantization** | precise_quantization_type | string | "fp32" | Conditional | Precise quantization type used for reorder when `use_reorder=true` and `reorder_source="precise"` |
| **Graph** | max_degree | int | 64 | No | Max edges per node |
| **Graph** | ef_construction | int | 400 | No | Candidate list size during construction |
| **Graph** | graph_type | string | "nsw" | No | Graph algorithm: nsw, odescent |
| **Memory** | hgraph_init_capacity | int | 100 | No | Initial index capacity |
| **Performance** | build_thread_count | int | 100 | No | Construction thread count |
| **Storage** | base_io_type | string | "block_memory_io" | No | Base quantization storage type |
| **Storage** | base_file_path | string | "./default_file_path" | No | Base quantization file path |
| **Storage** | precise_io_type | string | "block_memory_io" | No | Precise quantization storage type |
| **Storage** | precise_file_path | string | "./default_file_path" | No | Precise quantization file path |
| **Advanced** | base_pq_dim | int | 128 | Conditional | PQ subspace count |
| **Advanced** | ignore_reorder | bool | false | No | Skip precise quantization serialization |
| **Advanced** | build_by_base | bool | false | No | Build index using base quantization |
| **Features** | support_duplicate | bool | false | No | Enable duplicate data detection |
| **Features** | duplicate_distance_threshold | float | 0.0 | No | Deduplicate by nearest-candidate distance when greater than 0; otherwise fall back to code memcmp |
| **Features** | support_remove | bool | false | No | Enable graph delete-tracking metadata |
| **Features** | support_force_remove | bool | false | No | Enable force-remove support and its extra synchronization |
| **Features** | store_raw_vector | bool | false | No | Store raw vectors (cosine metric) |
| **Features** | use_elp_optimizer | bool | false | No | Auto parameter optimization |

## Detailed Explanation of Building Parameters

### dtype
- **Parameter Type**: string
- **Parameter Description**: Data type for vector elements
- **Optional Values**: "float32" (currently only supports float32)
- **Default Value**: "float32"

### metric_type
- **Parameter Type**: string
- **Parameter Description**: Distance metric type for similarity calculation
- **Optional Values**: "l2", "ip", "cosine"
- **Default Value**: "l2"

### dim
- **Parameter Type**: int
- **Parameter Description**: Vector dimension
- **Optional Values**: 1 to 4096
- **Default Value**: Must be provided (no default value)

### base_quantization_type
- **Parameter Type**: string
- **Parameter Description**: Base quantization type for vector compression
- **Optional Values**: "fp32", "fp16", "bf16", "sq8", "sq8_uniform", "sq4_uniform", "pq", "rabitq", "pqfs"
- **Default Value**: Must be provided (no default value)

### base_codes_type
- **Parameter Type**: string
- **Parameter Description**: Base code storage type. Set to "rabitq_split" together with base_quantization_type="rabitq" and rabitq_version="split_1bit_7bit" to use RabitQ split 1bit + 7bit storage.
- **Optional Values**: "flatten", "rabitq_split"
- **Default Value**: "flatten"

### rabitq_version
- **Parameter Type**: string
- **Parameter Description**: RabitQ implementation version. The default keeps the existing RabitQ path; "split_1bit_7bit" enables the opt-in split storage path.
- **Optional Values**: "standard", "split_1bit_7bit"
- **Default Value**: "standard"

### rabitq_error_rate
- **Parameter Type**: float
- **Parameter Description**: Error-rate multiplier for RabitQ one-bit lower-bound search. This parameter is stored with the index and must match when loading a serialized index.
- **Optional Values**: finite positive float
- **Default Value**: 1.9

### use_reorder
- **Parameter Type**: bool
- **Parameter Description**: Whether to reorder/rerank search results after base graph search. The reorder source is controlled by `reorder_source`.
- **Optional Values**: true, false
- **Default Value**: false

### reorder_source
- **Parameter Type**: string
- **Parameter Description**: Source codes used for reorder when `use_reorder=true`. Set to "precise" to use precise reorder codes, or "base" to reuse base codes for reorder. With `reorder_source="base"`, HGraph does not require `precise_quantization_type` or precise reorder codes.
- **Optional Values**: "precise", "base"
- **Default Value**: "precise"

### precise_quantization_type
- **Parameter Type**: string
- **Parameter Description**: Precise quantization type used for reordering, only effective when `use_reorder=true` and `reorder_source="precise"`
- **Optional Values**: "fp32", "fp16", "bf16", "sq8", "sq8_uniform", "sq4_uniform", "pq", "rabitq", "pqfs"
- **Default Value**: "fp32"

### max_degree
- **Parameter Type**: int
- **Parameter Description**: Maximum degree (number of edges) per node in the graph
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 64

### ef_construction
- **Parameter Type**: int
- **Parameter Description**: Size of the dynamic candidate list during graph construction, affects construction quality
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 400

### hgraph_init_capacity
- **Parameter Type**: int
- **Parameter Description**: Initial capacity when creating the index (not the actual size)
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 100

### build_thread_count
- **Parameter Type**: int
- **Parameter Description**: Number of threads used for index construction
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 100

### graph_type
- **Parameter Type**: string
- **Parameter Description**: Graph construction algorithm type
- **Optional Values**: "nsw", "odescent"
- **Default Value**: "nsw"

### base_io_type
- **Parameter Type**: string
- **Parameter Description**: Storage type for base quantization codes
- **Optional Values**: "memory_io", "block_memory_io", "buffer_io", "async_io", "mmap_io"
- **Default Value**: "block_memory_io"

### base_file_path
- **Parameter Type**: string
- **Parameter Description**: File path for base quantization codes storage, only meaningful when base_io_type is not memory-based
- **Optional Values**: Any valid file path
- **Default Value**: "./default_file_path"

### precise_io_type
- **Parameter Type**: string
- **Parameter Description**: Storage type for precise quantization codes, same as base_io_type but for reordering codes
- **Optional Values**: "memory_io", "block_memory_io", "buffer_io", "async_io", "mmap_io"
- **Default Value**: "block_memory_io"

### precise_file_path
- **Parameter Type**: string
- **Parameter Description**: File path for precise quantization codes storage
- **Optional Values**: Any valid file path
- **Default Value**: "./default_file_path"

### ignore_reorder
- **Parameter Type**: bool
- **Parameter Description**: Whether to ignore precise quantization during serialization
- **Optional Values**: true, false
- **Default Value**: false

### build_by_base
- **Parameter Type**: bool
- **Parameter Description**: Whether to build the index using base quantization codes instead of precise codes
- **Optional Values**: true, false
- **Default Value**: false

### base_pq_dim
- **Parameter Type**: int
- **Parameter Description**: Number of subspaces for PQ quantization, required when base_quantization_type is "pq" or "pqfs"
- **Optional Values**: 1 to dim
- **Default Value**: 128

### support_duplicate
- **Parameter Type**: bool
- **Parameter Description**: Whether to enable duplicate data detection to reduce the impact of duplicate vectors
- **Optional Values**: true, false
- **Default Value**: false

### duplicate_distance_threshold
- **Parameter Type**: float
- **Parameter Description**: Duplicate-detection distance threshold. When greater than 0, the nearest candidate is treated as the duplicate owner if its distance is within the threshold; when 0, duplicate detection falls back to code memcmp with the nearest candidate
- **Optional Values**: Any non-negative float
- **Default Value**: 0.0

### store_raw_vector
- **Parameter Type**: bool
- **Parameter Description**: Whether to store raw vectors in the index, useful for cosine metric
- **Optional Values**: true, false
- **Default Value**: false

### use_elp_optimizer
- **Parameter Type**: bool
- **Parameter Description**: Whether to automatically optimize internal parameters after construction based on system conditions
- **Optional Values**: true, false
- **Default Value**: false

### support_remove
- **Parameter Type**: bool
- **Parameter Description**: Whether to enable graph delete-tracking metadata
- **Optional Values**: true, false
- **Default Value**: false

### support_force_remove
- **Parameter Type**: bool
- **Parameter Description**: Whether to enable the force-remove path and its extra synchronization
- **Optional Values**: true, false
- **Default Value**: false

## Examples for Build Parameter String
```json
"index_param": {
    "base_quantization_type": "sq8",
    "max_degree": 32,
    "ef_construction": 200
}
```
means that the index is built using SQ8 quantization, with a maximum degree of 32 and ef_construction of 200.

```json
"index_param": {
    "base_quantization_type": "pq",
    "base_pq_dim": 64,
    "use_reorder": true,
    "precise_quantization_type": "fp16",
    "max_degree": 64,
    "ef_construction": 400,
    "build_thread_count": 50,
    "support_duplicate": true,
    "duplicate_distance_threshold": 0.02,
    "support_force_remove": true
}
```
means that the index uses PQ quantization with 64 subspaces, enables reordering with FP16 precision, deduplicates inserts within distance threshold 0.02, and enables force-remove support with maximum degree 64 and ef_construction 400.

## Detailed Explanation of Search Parameters

### ef_search
- **Parameter Type**: int
- **Parameter Description**: Size of the dynamic candidate list during search, affects search quality and speed
- **Optional Values**: 1 to INT_MAX
- **Default Value**: Must be provided (no default value)

### rabitq_one_bit_search
- **Parameter Type**: bool
- **Parameter Description**: Whether to use the RabitQ split one-bit search path when the index was built with base_codes_type="rabitq_split" and rabitq_version="split_1bit_7bit".
- **Optional Values**: true, false
- **Default Value**: false
- **Note**: This mode supports the `parallelism` search parameter for parallel search within a single query.

### brute_force_threshold
- **Parameter Type**: float
- **Parameter Description**: Selectivity-aware brute-force fallback. When set to a value greater than `0.0` and the active filter's `ValidRatio()` is less than or equal to this threshold, the search bypasses graph traversal and runs an exact scan over the valid ids using the best available flatten codes (raw vectors > precise reorder codes > base quantized codes, in that order of preference). The post-search reorder pass is skipped on queries that take the brute-force branch.
- **Optional Values**: any float in `[0.0, 1.0]`
- **Default Value**: 0.0 (disabled — preserves legacy behavior)
- **Applies to**: `KnnSearch` (non-iterator overload, also used by `SearchWithRequest`) and `RangeSearch`. The iterator-style `KnnSearch` does not use this parameter.
- **Note**: The decision relies on `Filter::ValidRatio()` returning a meaningful selectivity estimate; see [filtered search](docs/docs/en/src/advanced/filtered_search.md). The brute-force scan visits every indexed id once to call `CheckValid`, so its cost is roughly `O(N × dim)` regardless of selectivity. A runnable example is [`322_feature_hgraph_brute_force_threshold.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_hgraph_brute_force_threshold.cpp).

## Examples for Search Parameter String
```json
"hgraph": {
    "ef_search": 200,
    "parallelism": 4,
    "rabitq_one_bit_search": true
}
```
means that the search will use an ef_search value of 200 to control the search quality and performance trade-off. When the index uses RabitQ split storage, rabitq_one_bit_search=true enables one-bit graph search and parallelism=4 enables parallel search within a single query.

```json
"hgraph": {
    "ef_search": 200,
    "brute_force_threshold": 0.02
}
```
means that whenever the request supplies a filter whose `ValidRatio()` is ≤ 0.02 (i.e. only ~2% of the indexed ids survive the predicate), HGraph will skip the graph traversal and run an exact scan over the surviving ids; queries with weaker filters or no filter at all keep using the normal graph search.
