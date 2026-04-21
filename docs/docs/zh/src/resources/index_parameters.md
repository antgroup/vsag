# 索引参数

本页汇总 VSAG 各索引类型的常用参数。完整枚举请参考源码：

- 构建参数键：`src/constants.cpp`
- 公开常量：`include/vsag/constants.h`
- 每个索引的示例：`examples/cpp/101_index_hnsw.cpp` 等

## 通用参数

所有索引在构建时都需要提供以下顶层字段：

| 字段 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | 向量维度，构建后不可更改 |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | 向量数据类型，决定索引内部表示 |
| `metric_type` | `l2` / `ip` / `cosine` | 距离度量 |

## HNSW

HNSW 使用 `hnsw` 子对象承载构建参数，并不支持 HGraph 专有参数（如 `base_quantization_type`）。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "hnsw": {
        "max_degree": 32,
        "ef_construction": 400,
        "use_conjugate_graph": false
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `max_degree` | 16~48 | 每节点最大出边数 |
| `ef_construction` | 200~500 | 构建阶段候选集大小，越大召回越高、构建越慢 |
| `use_conjugate_graph` | bool | 是否构建 [共轭图](../advanced/enhance_graph.md) |

搜索时：

```json
{"hnsw": {"ef_search": 100, "use_conjugate_graph_search": false}}
```

## HGraph

HGraph 的构建参数使用通用的 `index_param` 键（参见 `examples/cpp/103_index_hgraph.cpp`）；
`hgraph` 键则保留给搜索期参数。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `max_degree` | 16~48 | 每节点最大出边数 |
| `ef_construction` | 200~500 | 构建阶段候选集大小，越大召回越高、构建越慢 |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | 主存储的量化策略 |

搜索时：

```json
{"hgraph": {"ef_search": 100}}
```

## DiskANN

```json
{
    "diskann": {
        "max_degree": 32,
        "ef_construction": 400,
        "pq_sample_rate": 0.1,
        "pq_dims": 32,
        "use_async_io": true
    }
}
```

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

无需额外参数。

## Pyramid

Pyramid 支持按 tag 组织多棵子图：

```json
{
    "pyramid": {
        "tag_dim": 1,
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

## SINDI（稀疏向量）

```json
{
    "sindi": {
        "top_k": 32,
        "doc_prune_ratio": 0.1
    }
}
```

## 运行期参数

除构建参数外，`Index::Tune` 与 `SearchParam` 可在运行时调整 `ef_search`、`nprobe` 等参数。参考
[优化器](../advanced/optimizer.md) 与各 `examples/cpp/3xx_feature_*.cpp` 示例。
