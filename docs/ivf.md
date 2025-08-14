# IVF Index

## Definition
IVF (Inverted File Index) improves search efficiency by **partitioning** data into buckets, thus reducing the search scope.

## Working Principle
1. **Clustering Phase**:
    First, perform a clustering operation on the entire high - dimensional vector dataset, dividing it into multiple non - overlapping clusters (also known as inverted lists). Commonly used clustering algorithms include K - means, etc. The cluster centers are called centroids. Suppose there are $n$ vectors clustered into $m$ clusters, with each cluster having a centroid.
2. **Index Building Phase**:
    Each vector is assigned to the cluster whose centroid is closest to it. The vector's information (such as the vector ID) is added to the corresponding inverted list. In this way, the IVF index is built, with each inverted list containing all the vectors belonging to that cluster.
3. **Search Phase**:
    When a query vector is given, first calculate the distances between the query vector and all centroids, and find the $k$ closest centroids ($k$ is a retrieval parameter). Then, perform an exact nearest - neighbor search only within the inverted lists corresponding to these $k$ centroids, significantly reducing the number of vectors to be searched.

## Suitable Scenarios (Recommended for use if any 2 - 4 of the following conditions are met)
1. The vector dimension is not very high, usually less than 512 dimensions. High dimensions may lead to the "curse of dimensionality" problem (disadvantage).
2. High - scale data scenarios, typically with over 100 million data points.
3. Memory - constrained scenarios, as its memory usage is lower than that of graph algorithms.
4. Large top - k recall requirements or complex filtering scenarios.

## Usage
For examples, refer to [106_index_ivf.cpp](https://github.com/antgroup/vsag/blob/main/examples/cpp/106_index_ivf.cpp).

## Detailed Explanation of Building Parameters

### partition_strategy_type
- **Parameter Type**: string
- **Parameter Description**: Bucket partitioning strategy type
- **Optional Values**: "ivf", "gno_imi"
- **Default Value**: "ivf"

### first_order_buckets_count
- **Parameter Type**: int
- **Parameter Description**: Only effective when `partition_strategy_type` is "gno_imi", representing the number of first - level buckets.
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 10

### second_order_buckets_count
- **Parameter Type**: int
- **Parameter Description**: Only effective when `partition_strategy_type` is "gno_imi", representing the number of second - level buckets.
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 10

### buckets_count
- **Parameter Type**: int
- **Parameter Description**: Only effective when `partition_strategy_type` is "ivf", representing the number of buckets.
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 10

### ivf_train_type
- **Parameter Type**: string
- **Parameter Description**: Clustering algorithm type
- **Optional Values**: "kmeans", "random"
- **Default Value**: "kmeans"

### base_quantization_type
- **Parameter Type**: string
- **Parameter Description**: Coarse - ranking vector quantization type (encoding of in - bucket vectors)
- **Optional Values**: "fp32", "fp16", "bf16", "sq8", "sq8_uniform", "sq4_uniform", "pq", "rabitq", "pqfs"
- **Default Value**: "fp32"

### base_io_type
- **Parameter Type**: string
- **Parameter Description**: Coarse - ranking vector IO type (storage access type of in - bucket vectors)
- **Optional Values**: "memory_io", "block_memory_io"
- **Default Value**: "memory_io"

### base_pq_dim
- **Parameter Type**: int
- **Parameter Description**: Coarse - ranking vector PQ dimension, used for re - ranking
- **Optional Values**: 1 to dim
- **Default Value**: 1

### use_reorder
- **Parameter Type**: bool
- **Parameter Description**: Whether to use re - ranking
- **Optional Values**: true, false
- **Default Value**: false

### precise_quantization_type
- **Parameter Type**: string
- **Parameter Description**: Fine - ranking vector quantization type, used for re - ranking
- **Optional Values**: "fp32", "fp16", "bf16", "sq8", "sq8_uniform", "sq4_uniform", "pq", "rabitq", "pqfs"
- **Default Value**: "fp32"

### precise_io_type
- **Parameter Type**: string
- **Parameter Description**: Fine - ranking vector IO type, used for re - ranking
- **Optional Values**: "memory_io", "block_memory_io", "mmap_io", "buffer_io", "async_io", "reader_io"
- **Default Value**: "block_memory_io"

### precise_file_path
- **Parameter Type**: string
- **Parameter Description**: Fine - ranking vector file path, used for re - ranking
- **Optional Values**: Any valid file path
- **Default Value**: ""

## Examples for Build Parameter String
```json
"index_param": {
    "buckets_count": 50,
    "base_quantization_type": "fp32",
    "partition_strategy_type": "ivf",
    "ivf_train_type": "kmeans"
}
```
means that the index is built using 50 buckets, the base quantization type is fp32, the partition strategy type is ivf, and the ivf train type is kmeans.

```json
"index_param": {
    "buckets_count": 50,
    "base_quantization_type": "pqfs",
    "partition_strategy_type": "ivf",
    "ivf_train_type": "random",
    "precise_quantization_type": "fp16",
    "use_reorder": true,
    "base_pq_dim": 32,
    "precise_io_type": "async_io",
    "precise_file_path": "./precise_codes"
}
```
means that the index is built using 50 buckets, the base quantization type is pqfs with pq dim = 32, the partition strategy type is ivf, and the ivf train type is random. this configuration enables reordering, the precise quantization type is fp16, uses libaio's asynchronous I/O for precise operations, and specifies the file for precise codes as './precise_codes'

## Detailed Explanation of Search Parameters

### scan_buckets_count
- **Parameter Type**: int
- **Parameter Description**: Number of buckets to scan
- **Optional Values**: 1 to buckets_count
- **Default Value**: **must be provided (no default value)**

### factor
- **Parameter Type**: float
- **Parameter Description**: Scan factor, used for reordering, for example, if topk=10, factor=2.0, then IVF stage will recall 20 points, and then use precise code for reordering
- **Optional Values**: 1.0 to FLOAT_MAX
- **Default Value**: 2.0

### parallelism
- **Parameter Type**: int
- **Parameter Description**: Number of threads to use for parallel search per query
- **Optional Values**: 1 to INT_MAX
- **Default Value**: 1 (only the search main thread do the search)

### timeout_ms
- **Parameter Type**: double
- **Parameter Description**: Maximum time cost in milliseconds for each query, used to control the search time cost
- **Optional Values**: 1 to DOUBLE_MAX
- **Default Value**: DOUBLE_MAX

## Examples for Search Parameter String
```json
"ivf": {
    "scan_buckets_count": 10,
    "factor": 2.0,
    "parallelism": 4,
    "timeout_ms": 30.0
}
```
means that the search will scan 10 buckets, the factor is 2.0, and the parallelism is 4, around 4 threads per query, and the max time cost is 30ms (when search time exceed 30ms, will return the current result). 
