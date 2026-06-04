为我写一个测试脚本测试dmq算法的性能，测试的数据集是/root/vsag/data/wholenet_10M/wholenet-10m.hdf5
构建和查询参数如下，你用eval_performance测评。
测评开启dmq和不开启dmq的对比，包含qps和search memory peak, index size，再单独计算rerank正排数据size大小
最后整理成实md文档的实验报告
BUILD_PARAMETER='{
  "dtype": "sparse",
  "dim": 301,
  "metric_type": "ip",
  "index_param": {
    "term_id_limit": 300000,
    "window_size": 50000,
    "doc_prune_ratio": 0.2,
    "use_quantization": true,
    "use_reorder": true,
    "rerank_type": "dmq",
    "dmq_bits": 8,
    "avg_doc_term_length": 50
  }
}'

SEARCH_PARAMETER='{
  "sindi": {
    "query_prune_ratio": 0.2,
    "term_prune_ratio": 0,
    "n_candidate": 200,
    "use_term_lists_heap_insert": true
  }
}'