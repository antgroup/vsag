# VSAG Benchmarks

## Benchmark Scripts

### SINDI SQ8 + Reorder Comparison

The `run_sindi_sq8_reorder_compare.sh` script compares two reorder variants for SINDI sparse vectors:
- **SQ8 + FP32 Reorder:** High-precision reranking using full-precision (FP32) forward scores
- **SQ8 + DMQ8 Reorder:** Fast approximate reranking using 8-bit quantized codes

Both variants share the same SQ8 quantization for the inverted posting-list phase (`use_quantization: true`),
but differ in the reorder backend (rerank phase). This allows fair comparison of memory vs. accuracy trade-offs.

#### Usage

```bash
DATAPATH=/path/to/sparse.hdf5 DIM=512 TOPK=10 N_CANDIDATE=200 \
	NUM_THREADS_SEARCHING=32 ./benchs/run_sindi_sq8_reorder_compare.sh
```

#### Configuration

Key environment variables:
- `DATAPATH` (required): Path to sparse HDF5 dataset
- `DIM`: Vector dimension (default: 512)
- `TOPK`: Number of nearest neighbors to retrieve (default: 10)
- `N_CANDIDATE`: Candidates to rerank after inverted phase (default: 200)
- `QUERY_PRUNE_RATIO`: Fraction of query terms to prune (default: 0.2)
- `TERM_PRUNE_RATIO`: Fraction of posting-list terms to prune (default: 0.1)
- `USE_QUANTIZATION`: Enable SQ8 quantization (always true in this script)
- `NUM_THREADS_BUILDING`, `NUM_THREADS_SEARCHING`: Thread count for build/search phases

#### Output

For each variant, the script generates:
- JSON results with metrics (latency, memory, recall)
- Markdown table with searchable summaries
- Index file for reuse

## Recall 90%



## Recall 90%

|             | HNSW                                                         | DiskANN                                                      | HGraph                                                       | IVF                                                          |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DEEP1B-96   |                                                              |                                                              | {<br/>  "dim": 96,<br/>  "dtype": "float32",<br/>  "metric_type": "cosine",<br/>  "index_param": {<br/>    "base_quantization_type": "fp32",<br/>    "max_degree": 48,<br/>    "graph_type": "odescent",<br/>    "alpha": 1.2,<br/>    "graph_iter_turn": 40,<br/>    "neighbor_sample_rate": 0.2<br/>  }<br/>} |                                                              |
| GloVe-100   |                                                              |                                                              | {<br/>  "dim": 100,<br/>  "dtype": "float32",<br/>  "metric_type": "cosine",<br/>  "index_param": {<br/>    "base_quantization_type": "fp32",<br/>    "graph_type": "odescent",<br/>    "max_degree": 96,<br/>    "alpha": 1.2,<br/>    "graph_iter_turn": 30,<br/>    "neighbor_sample_rate": 0.2<br/>  }<br/>} |                                                              |
| SIFT-128    | {<br/>  "dim": 128,<br/>  "dtype": "float32",<br/>  "metric_type": "l2",<br/>  "hnsw": {<br/>    "max_degree": 12,<br/>    "ef_construction": 500<br/>  }<br/>} | {<br/>  "dim": 128,<br/>  "dtype": "float32",<br/>  "metric_type": "l2",<br/>  "diskann": {<br/>    "max_degree": 16,<br/>    "ef_construction": 200,<br/>    "pq_sample_rate": 0.5,<br/>    "pq_dims": 16,<br/>    "use_pq_search": true,<br/>    "use_bsa": true<br/>  }<br/>} | {<br/>  "dim": 128,<br/>  "dtype": "float32",<br/>  "metric_type": "l2",<br/>  "index_param": {<br/>    "base_quantization_type": "sq8_uniform",<br/>    "graph_type": "odescent",<br/>    "max_degree": 64,<br/>    "alpha": 1.2,<br/>    "graph_iter_turn": 50,<br/>    "neighbor_sample_rate": 0.2<br/>  }<br/>} | {<br/>  "dim": 128,<br/>  "dtype": "float32",<br/>  "metric_type": "l2",<br/>  "index_param": {<br/>    "buckets_count": 1000,<br/>    "base_quantization_type": "fp32",<br/>    "partition_strategy_type": "ivf",<br/>    "ivf_train_type": "kmeans"<br/>  }<br/>} |
| NYTimes-256 |                                                              |                                                              | {<br/>  "dim": 256,<br/>  "dtype": "float32",<br/>  "metric_type": "cosine",<br/>  "index_param": {<br/>    "base_quantization_type": "sq8_uniform",<br/>    "max_degree": 48,<br/>    "ef_construction": 400<br/>  }<br/>} |                                                              |
| COHERE-768  |                                                              |                                                              | {<br/>  "dim": 768,<br/>  "dtype": "float32",<br/>  "metric_type": "cosine",<br/>  "index_param": {<br/>    "base_quantization_type": "sq8_uniform",<br/>    "max_degree": 64,<br/>    "ef_construction": 400,<br/>    "precise_quantization_type": "fp32",<br/>    "use_reorder": true<br/>  }<br/>} |                                                              |
| GIST-960    |                                                              |                                                              | {<br/>  "dim": 960,<br/>  "dtype": "float32",<br/>  "metric_type": "l2",<br/>  "index_param": {<br/>    "base_quantization_type": "sq8_uniform",<br/>    "max_degree": 64,<br/>    "graph_type": "odescent",<br/>    "alpha": 1.2,<br/>    "graph_iter_turn": 60,<br/>    "neighbor_sample_rate": 0.2,<br/>    "precise_quantization_type": "fp32",<br/>    "use_reorder": true<br/>  }<br/>} |                                                              |
| OPENAI-1536 |                                                              |                                                              | {<br/>  "dim": 1536,<br/>  "dtype": "float32",<br/>  "metric_type": "cosine",<br/>  "index_param": {<br/>    "base_quantization_type": "fp32",<br/>    "max_degree": 64,<br/>    "graph_type": "odescent",<br/>    "alpha": 1.2,<br/>    "graph_iter_turn": 30,<br/>    "neighbor_sample_rate": 0.1<br/>  }<br/>} |                                                              |

