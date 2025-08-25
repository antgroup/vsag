# Index Analysis Report
**Index Name**: [index name]

---

## 1. **Index Configuration Overview**
- **Vector Dimension**: [dim value]
- **Data Type**: [float32/int8/...]
- **Distance Metric**: [l2/ip/...]
- **Index Type**: [HGraph/IVF/...]
- **Storage Backend**: [block_memory_io/memory_io/...]

---

## 2. **Index Inner Property Explanation**
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

## 3. **Search Analyze Explanation**
```json
{
    "avg_distance_query": "Average distance between query vectors and retrieved nearest neighbors",
    "quantization_bias_ratio": "Quantization bias observed during search phase",
    "quantization_inversion_count_rate": "Rate of distance inversions caused by quantization during search",
    "recall_query": "Recall rate of the search (proportion of true nearest neighbors retrieved)",
    "time_cost_query": "Average time cost per query in milliseconds"
}
```
