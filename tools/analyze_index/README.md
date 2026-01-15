# VSAG Index Analysis Tool (`analyze_index`)

[\[中文\]](README_zh.md)

`analyze_index` is a powerful command-line diagnostic tool designed for the VSAG library. It allows developers and advanced users to gain deep insights into the internal state, metadata, and performance characteristics of a pre-built VSAG index file.

The tool supports two core analysis modes:

1.  **Static Property Analysis**: Parses an index file directly without needing query data to display its basic information (e.g., dimension, metric type) and internal structural health metrics.
2.  **Dynamic Query Analysis**: When a query dataset is provided, it simulates the search process and analyzes the index's behavior, reporting dynamic performance metrics like recall and latency.

## Key Features

-   **Display Index Metadata**: Shows basic information such as the index's dimension, data type, and distance metric.
-   **Reveal Internal Structural Properties**: Outputs deep health metrics like connected components, duplicate vector rate, and quantization bias.
-   **Analyze Query Behavior**: Evaluates the index's query recall, average distance, and query latency on a given query set.
-   **Support for Multiple Index Types**: Currently supports analysis for mainstream index types like HGraph and IVF.
-   **Flexible Parameter Configuration**: Allows overriding the index's build and search parameters during analysis for "what-if" scenarios.

## Command-Line Arguments

| Argument               | Alias | Required | Description                                                                                                                                  |
| ---------------------- | ----- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--index_path`         | `-i`  | **Yes**  | The path to the VSAG index file to be analyzed.                                                                                              |
| `--build_parameter`    | `-bp` | No       | Build parameters (JSON string) to use when loading the index. If not provided, the original parameters stored in the index file will be used. |
| `--query_path`         | `-qp` | No       | Path to the query dataset for dynamic query analysis. If not provided, only static analysis will be performed.                               |
| `--search_parameter`   | `-sp` | No       | Search parameters (JSON string) to be used during the query.                                                                                 |
| `--topk`               | `-k`  | No       | The number of top K results to return during the query (default: 100).                                                                       |

## Usage Examples

#### Example 1: Static Analysis Only

Analyze the index file itself without performing any queries.

```bash
./analyze_index --index_path /path/to/my_index.hnsw
```

**Sample Output (Partial):**

```
[info] index name: hnsw
[info] index dim: 128
[info] index data type: float
[info] index metric: l2
[info] index param: {
    "M": 32,
    "ef_construction": 400
}
[info] index inner property: {
    "connect_components": 1,
    "deleted_count": 0,
    ...
}
```

#### Example 2: Static and Dynamic Query Analysis

Analyze the index file and use a query file to evaluate its search behavior.

```bash
./analyze_index \
    --index_path /path/to/my_index.ivf \
    --query_path /path/to/queries.bin \
    --search_parameter '{"nprobe": 16}' \
    --topk 50
```

**Sample Output (Partial):**
In addition to the static information above, it will also output:

```
[info] Search Analyze: {
    "avg_distance_query": 0.45,
    "recall_query": 0.98,
    "time_cost_query": 1.2
    ...
}
```

## Output Metrics Explained

### 1. Index Configuration Overview

| Property             | Description                             |
| -------------------- | --------------------------------------- |
| **Vector Dimension** | The dimension of the vectors.           |
| **Data Type**        | The data type (float32/int8...).        |
| **Distance Metric**  | The distance metric (l2/ip...).         |
| **Index Type**       | The index algorithm (HGraph/IVF...).    |

### 2. Index Inner Properties (`GetStats`)

These metrics reflect the static health and state of the index after it has been built.

```json
{
    "avg_distance_base": "Average distance between vectors in the base dataset (pre-indexing)",
    "connect_components": "Number of connected components in the index graph structure",
    "deleted_count": "Number of vectors marked for deletion in the index",
    "duplicate_rate": "Proportion of duplicate vectors in the dataset",
    "proximity_recall_neighbor": "Recall rate for neighbor proximity verification in the index",
    "quantization_bias_ratio": "Ratio representing quantization bias in compressed vector representation",
    "quantization_inversion_count_rate": "Rate of quantization-induced distance inversions (incorrect orderings)",
    "recall_base": "Recall rate of the base dataset (ground truth for comparison)",
    "total_count": "Total number of vectors in the index"
}
```

### 3. Search Analysis (`AnalyzeIndexBySearch`)

These metrics reflect the dynamic performance of the index when handling query requests.

```json
{
    "avg_distance_query": "Average distance between query vectors and retrieved nearest neighbors",
    "quantization_bias_ratio": "Quantization bias observed during the search phase",
    "quantization_inversion_count_rate": "Rate of distance inversions caused by quantization during search",
    "recall_query": "Recall rate of the search (proportion of true nearest neighbors retrieved)",
    "time_cost_query": "Average time cost per query in milliseconds"
}
```

## License

This tool is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
