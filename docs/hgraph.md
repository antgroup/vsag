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

## Factory Parameter Overview Table

| **Category** | **Parameter** | **Type** | **Default Value** | **Required** | **Description** |
|--------------|---------------|----------|-------------|--------------|-----------------|
| **Basic** | dtype | string | "float32" | Yes | Data type (only float32 supported) |
| **Basic** | metric_type | string | "l2" | Yes | Distance metric: l2, ip, cosine |
| **Basic** | dim | int | - | Yes | Vector dimension [1, 4096] |
| **Quantization** | base_quantization_type | string | - | Yes | Base quantization type |
| **Quantization** | use_reorder | bool | false | No | Enable high-precision reordering |
| **Quantization** | precise_quantization_type | string | "fp32" | Conditional | High-precision type for reordering |
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
| **Features** | support_remove | bool | false | No | Enable deletion support |
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

### use_reorder
- **Parameter Type**: bool
- **Parameter Description**: Whether to use high-precision quantization for reordering to improve accuracy
- **Optional Values**: true, false
- **Default Value**: false

### precise_quantization_type
- **Parameter Type**: string
- **Parameter Description**: High-precision quantization type used for reordering, only effective when use_reorder=true
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
- **Parameter Description**: Whether to support deletion operations
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
    "support_remove": true
}
```
means that the index uses PQ quantization with 64 subspaces, enables reordering with FP16 precision, supports duplicate detection and deletion, with maximum degree 64 and ef_construction 400.

## Detailed Explanation of Search Parameters

### ef_search
- **Parameter Type**: int
- **Parameter Description**: Size of the dynamic candidate list during search, affects search quality and speed
- **Optional Values**: 1 to INT_MAX
- **Default Value**: Must be provided (no default value)

## Examples for Search Parameter String
```json
"hgraph": {
    "ef_search": 200
}
```
means that the search will use an ef_search value of 200 to control the search quality and performance trade-off.