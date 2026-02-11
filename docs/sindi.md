# SINDI Index

## Definition

SINDI (Sparse Inverted Index) is an **inverted index** structure specifically designed for sparse vector approximate nearest neighbor search. It leverages the sparsity of vectors by organizing data in a term-based inverted list structure, achieving efficient similarity search on high-dimensional sparse data.

## Working Principle

### 1. Index Construction Phase

SINDI organizes sparse vectors using a **window-based inverted list structure**:

- **Window Partitioning**: The dataset is divided into fixed-size windows (controlled by `window_size` parameter). Each window maintains its own term lists.

- **Inverted List Structure**: For each window, SINDI builds inverted lists where each term (dimension) maintains a list of document IDs and their corresponding values. This allows efficient retrieval of documents containing specific terms.

- **Document Pruning**: During construction, documents can be pruned based on the `doc_prune_ratio` parameter to reduce storage overhead by removing low-importance terms.

- **Quantization Support**: Optional quantization (controlled by `use_quantization`) can compress term values to reduce memory usage.

- **Reordering Support**: When `use_reorder` is enabled, SINDI maintains a high-precision flat index for accurate distance recalculation during search.

### 2. Search Phase

When a query sparse vector is provided:

1. **Term-based Scoring**: The query vector's non-zero terms are used to score candidate documents by aggregating contributions from the inverted lists.

2. **Window Iteration**: SINDI iterates through relevant windows (optionally filtered by ID filters) and computes similarity scores.

3. **Heap-based Selection**: Top-k candidates are maintained using a max-heap structure for efficient retrieval.

4. **Reordering (Optional)**: If `use_reorder` is enabled, candidates are re-scored using high-precision calculations to improve accuracy.

5. **Result Return**: The final results are returned with distances converted from inner product scores (using `distance = 1 - inner_product`).

## Suitable Scenarios

1. **Sparse Vector Data**: Documents represented as sparse vectors (e.g., TF-IDF, BM25, neural sparse representations).

2. **High-dimensional Sparse Data**: Scenarios where vectors have very high dimensionality but most values are zero.

3. **Text Retrieval**: Semantic search and information retrieval tasks using sparse embeddings.

4. **Memory-Constrained Environments**: When quantization is enabled, memory usage is significantly reduced.

5. **Dynamic Datasets**: Supports incremental addition of documents after initial index construction.

## Usage

For examples, refer to [109_index_sindi.cpp](https://github.com/antgroup/vsag/blob/main/examples/cpp/109_index_sindi.cpp).

## Factory Parameter Overview Table

| **Category** | **Parameter** | **Type** | **Default Value** | **Required** | **Description** |
|--------------|---------------|----------|-------------------|--------------|-----------------|
| **Basic** | dtype | string | "sparse" | Yes          | Data type (must be "sparse") |
| **Basic** | metric_type | string | "ip" | Yes          | Distance metric (only "ip" supported) |
| **Basic** | dim | int | - | Yes          | Maximum number of non-zero elements per sparse vector (id-value pairs) |
| **Index** | term_id_limit | int | 1000000 | No           | Maximum term ID limit |
| **Index** | window_size | int | 50000 | No           | Number of vectors per window [10000, 60000] |
| **Index** | doc_prune_ratio | float | 0.0 | No           | Document pruning ratio [0.0, 0.5] |
| **Performance** | use_reorder | bool | false | No           | Enable high-precision reordering |
| **Memory** | use_quantization | bool | false | No           | Enable value quantization |
| **Advanced** | avg_doc_term_length | int | 100 | No           | Average document term length for memory estimation |

## Detailed Explanation of Building Parameters

### dtype
- **Parameter Type**: string
- **Parameter Description**: Data type for vector elements
- **Optional Values**: "sparse" (only sparse vectors supported)
- **Default Value**: "sparse"

### metric_type
- **Parameter Type**: string
- **Parameter Description**: Distance metric type for similarity calculation
- **Optional Values**: "ip" (only inner product supported)
- **Default Value**: "ip"

### dim
- **Parameter Type**: int
- **Parameter Description**: Maximum number of non-zero elements (id-value pairs) allowed in a single sparse vector. For example, if a sparse vector is `{0:0.1, 2:0.5, 177:0.8}`, it has 3 non-zero elements, so its dimension is 3. This parameter limits the maximum length of sparse vectors that can be added to the index. Note: This is different from `term_id_limit`, which limits the maximum term ID value (e.g., 177 in the example).
- **Optional Values**: Positive integer
- **Default Value**: Must be provided (no default value)

### term_id_limit
- **Parameter Type**: int
- **Parameter Description**: Maximum term ID that the index can handle, determines the size of term lists
- **Optional Values**: Positive integer
- **Default Value**: 10000000

### window_size
- **Parameter Type**: int
- **Parameter Description**: Number of vectors stored in each window, affects memory layout and search efficiency
- **Optional Values**: 10000 to 60000
- **Default Value**: 50000

### doc_prune_ratio
- **Parameter Type**: float
- **Parameter Description**: Ratio of low-importance terms to prune from documents during indexing
- **Optional Values**: 0.0 to 0.9
- **Default Value**: 0.0 (no pruning)

### use_reorder
- **Parameter Type**: bool
- **Parameter Description**: Whether to maintain a high-precision flat index for accurate distance recalculation
- **Optional Values**: true, false
- **Default Value**: false

### use_quantization
- **Parameter Type**: bool
- **Parameter Description**: Whether to quantize term values to reduce memory usage
- **Optional Values**: true, false
- **Default Value**: false

### avg_doc_term_length
- **Parameter Type**: int
- **Parameter Description**: Average number of non-zero terms per document, used for memory estimation
- **Optional Values**: Positive integer
- **Default Value**: 100

## Examples for Build Parameter String

```json
{
    "dim": 10000,
    "dtype": "sparse",
    "metric_type": "ip",
    "index_param": {
        "term_id_limit": 5000,
        "window_size": 10000,
        "doc_prune_ratio": 0.1,
        "use_reorder": false,
        "use_quantization": false
    }
}
```

This creates a SINDI index allowing sparse vectors with a term ID limit of 5000, maximal dimension of 10000, window size of 10000, and 10% document pruning.

```json
{
    "dim": 50000,
    "dtype": "sparse",
    "metric_type": "ip",
    "index_param": {
        "term_id_limit": 20000,
        "window_size": 50000,
        "doc_prune_ratio": 0.0,
        "use_reorder": true,
        "use_quantization": true,
        "avg_doc_term_length": 50
    }
}
```

This creates a SINDI index with reordering and quantization enabled for better accuracy and memory efficiency.

## Detailed Explanation of Search Parameters

### n_candidate
- **Parameter Type**: int
- **Parameter Description**: Number of candidate documents to retrieve during search
- **Optional Values**: 1 to AMPLIFICATION_FACTOR * k
- **Default Value**: 0 (must be provided)

### query_prune_ratio
- **Parameter Type**: float
- **Parameter Description**: Ratio of low-importance query terms to prune during search
- **Optional Values**: 0.0 to 0.9
- **Default Value**: 0.0 (no pruning)

### term_prune_ratio
- **Parameter Type**: float
- **Parameter Description**: Ratio of low-importance terms to prune from term lists during search
- **Optional Values**: 0.0 to 0.9
- **Default Value**: 0.0 (no pruning)

### use_term_lists_heap_insert
- **Parameter Type**: bool
- **Parameter Description**: Whether to use term-list-based heap insertion for potentially better performance
- **Optional Values**: true, false
- **Default Value**: true

## Examples for Search Parameter String

```json
{
    "sindi": {
        "n_candidate": 100,
        "query_prune_ratio": 0.0,
        "term_prune_ratio": 0.0
    }
}
```

This searches with 100 candidates and no pruning.

```json
{
    "sindi": {
        "n_candidate": 200,
        "query_prune_ratio": 0.1,
        "term_prune_ratio": 0.05,
        "use_term_lists_heap_insert": true
    }
}
```

This searches with 200 candidates, 10% query term pruning, 5% term list pruning, and optimized heap insertion.

## Supported Features

- **Build & Add**: Supports building index and incremental addition of documents
- **Search**: Supports KNN search and range search with ID filtering
- **Serialization**: Supports binary serialization/deserialization, file-based, and reader set-based
- **Concurrency**: Supports concurrent search and add operations
- **Distance Calculation**: Supports calculating distance by document ID
- **Memory Estimation**: Can estimate memory usage before building

## Notes

1. SINDI is specifically designed for sparse vectors and does not support dense vectors.

2. Only inner product (IP) metric is supported for similarity calculation.

3. **Important**: Understand the difference between `dim` and `term_id_limit`:
   - `dim`: Maximum number of non-zero elements (id-value pairs) in a single sparse vector. For example, `{0:0.1, 2:0.5, 177:0.8}` has `dim=3`.
   - `term_id_limit`: Maximum value of term IDs. In the same example, the maximum term ID is 177, so `term_id_limit` should be at least 178.

4. The `term_id_limit` should be set based on the maximum term ID in your dataset.

5. Larger `window_size` values generally improve search performance but increase memory usage per window.

6. When `use_reorder` is enabled, memory usage approximately doubles but accuracy improves significantly.

7. Quantization reduces memory usage but may slightly decrease accuracy.
