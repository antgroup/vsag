# 索引参数

本页汇总 VSAG 各索引类型的常用参数。完整枚举请参考源码：

- 构建参数键：`src/constants.cpp`
- 公开常量：`include/vsag/constants.h`
- 每个索引的示例：`examples/cpp/*_index_*.cpp`（例如 `103_index_hgraph.cpp`）

## 通用参数

所有索引在构建时都接受以下顶层字段。其中 `dtype`、`metric_type` 必填，`repr` 可选。
非稀疏数据必须提供 `dim`；`dtype: "sparse"` 省略 `dim` 时默认使用 `4096`：

| 字段 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | 向量维度，构建后不可更改 |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` / `sparse` | 标量值类型；`sparse` 为稀疏索引兼容取值 |
| `repr` | `dense` / `sparse` / `multi_vector` | 可选的数据布局；省略时，`dtype: "sparse"` 推断为 `sparse`，其他类型推断为 `dense` |
| `metric_type` | `l2` / `ip` / `cosine` | 距离度量 |

`dtype` 与 `repr` 描述不同维度：前者是标量编码，后者是记录布局。显式设置 `repr` 时，
`dtype: "sparse"` 要求 `repr: "sparse"`。受支持的多向量索引使用
`repr: "multi_vector"`，并配合 `float32` 等标量 `dtype`。

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
| `use_reverse_edges` | `false` | 跟踪入边，实现 O(1) 反向邻居查找；边存储约翻倍，且压缩图存储不支持 |
| `label_remap_type` | `pg` | label map 实现：默认 `pg`，或 `robin` |
| `reorder_source` | `precise` | 从 `precise` 存储或直接从 `base` 重排；RaBitQ x+y split 会自动选择 `base` |
| `persist_source_id` | `false` | 序列化 HGraph 时保留 Source ID 元数据；适用于恢复索引后继续导出构建缓存 |
| `mrle_dim` | `0` | MRLE 输出维度，范围 `[0, dim]`；`0` 表示输入维度 |
| `fast_encode_rabitq` | `true` | 使用多 bit RaBitQ 快速编码；设为 `false` 恢复精确编码器 |
| `fast_encode_rabitq_rounds` | `6` | 快速编码器微调轮数，范围 `[1, 32]` |

搜索时：

```json
{"hgraph": {"ef_search": 100}}
```

`ef_search` 接受任意正的有符号 64 位整数，不再存在与 `topk` 相关的上限。非常大的取值会
明显增加延迟和搜索前沿占用的内存。

`hgraph` 搜索参数还接受 `brute_force_threshold`（`[0.0, 1.0]` 区间的 float，
默认 `0.0`）。当取值 `> 0` 且当前请求的 filter 的 `ValidRatio()` 不超过该
阈值时，HGraph 会跳过图遍历，直接在通过过滤的 id 上做精确暴扫。详见
[HGraph 索引文档](../indexes/hgraph.md#高选择性过滤下的暴搜回退brute_force_threshold)。

## LazyHGraph

LazyHGraph 的构建参数可以放在顶层 `lazy_hgraph` 对象中（推荐，语义更清晰），也可以放在
通用的 `index_param` 对象中。`hgraph` 子对象会转交给转换后的内部 HGraph。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `transition_threshold` | `1000` 或按业务规模设置 | 从精确 flat 搜索转换到 HGraph 的正整数向量数量阈值 |
| `hgraph` | HGraph 构建对象 | graph 阶段的参数；见 [HGraph](../indexes/hgraph.md) |

LazyHGraph 只支持 `dtype: "float32"`。搜索参数使用 `hgraph` 对象，例如
`{"hgraph": {"ef_search": 100}}`。详见 [LazyHGraph 索引文档](../indexes/lazy_hgraph.md)。

`hgraph` 搜索参数还接受以下 filter 相关参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_ratio` | float | `0.2` | 控制带 filter 搜索时跳过候选检查的比例，取值范围为 `[0.0, 1.0]`。值越大，跳过越激进，搜索越快但可能影响召回。 |
| `skip_strategy` | string | `"deterministic_accumulative"` | 跳过策略。支持 `"random"` 和 `"deterministic_accumulative"`。 |

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

Pyramid 构建参数同样放在 `index_param` 下：

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

## SINDI（稀疏向量）

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "doc_prune_ratio": 0.1
    }
}
```

`use_quantization`、不可变构建与 `n_candidate` 等搜索参数见
[SINDI 页面](../indexes/sindi.md)。

## 运行期参数

除构建参数外，`Index::Tune` 与 `SearchParam` 可在运行时调整 `ef_search`、`nprobe` 等参数。参考
[优化器](../advanced/optimizer.md) 与各 `examples/cpp/3xx_feature_*.cpp` 示例。
