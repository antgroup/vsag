# 索引参数

本页汇总 VSAG 各索引类型的常用参数。完整枚举请参考源码：

- 构建参数键：`src/constants.cpp`
- 公开常量：`include/vsag/constants.h`
- 每个索引的示例：`examples/cpp/*_index_*.cpp`（例如 `103_index_hgraph.cpp`）

## 通用参数

所有索引在构建时都需要提供以下顶层字段：

| 字段 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | 向量维度，构建后不可更改 |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | 向量数据类型，决定索引内部表示 |
| `metric_type` | `l2` / `ip` / `cosine` | 距离度量 |

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
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | 主存储的量化策略 —— 支持的全部取值见[量化章节](../quantization/README.md) |

搜索时：

```json
{"hgraph": {"ef_search": 100}}
```

`hgraph` 搜索参数还接受 `brute_force_threshold`（`[0.0, 1.0]` 区间的 float，
默认 `0.0`）。当取值 `> 0` 且当前请求的 filter 的 `ValidRatio()` 不超过该
阈值时，HGraph 会跳过图遍历，直接在通过过滤的 id 上做精确暴扫。详见
[HGraph 索引文档](../indexes/hgraph.md#高选择性过滤下的暴搜回退brute_force_threshold)。

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
