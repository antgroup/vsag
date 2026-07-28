# SINDI_V2

SINDI_V2 是 VSAG 的 term-first 稀疏向量索引。它与
[SINDI](sindi.md) 使用相同的窗口化倒排检索模型和 `1 - inner_product`
距离定义，但按 term 保存 posting，因此搜索时只需读取查询实际使用的倒排链。

- 工厂入口：`sindi_v2`
- 源码：`src/algorithm/sindi_v2/`
- `dtype` 必须为：`sparse`
- `metric_type` 必须为：`ip`

## 存储布局

每个 term payload 只保存非空 window：

```text
non_empty_window_count
ordered {window_id, posting_count}[]
local_doc_ids[]
alignment padding
encoded_values[]
```

查询时，SINDI_V2 将稀疏 metadata 展开为 `window_count + 1` 个 window offset，
并通过 `reader_io`、`mmap_io`、`buffer_io`、`async_io` 或内存后端，只加载
查询保留 term 对应的 posting。

这与 SINDI 的 window-first 序列化不同。两个工厂入口使用不同的持久化格式，
加载索引时不能互换。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "window_size": 50000,
        "doc_prune_ratio": 0.1,
        "use_quantization": true,
        "use_reorder": true,
        "remap_term_ids": true,
        "term_io": {
            "type": "buffer_io",
            "file_path": "/tmp/sindi_v2.terms"
        },
        "rerank_io": {
            "type": "block_memory_io"
        }
    }
})";
auto index = vsag::Factory::CreateIndex("sindi_v2", params).value();
index->Build(base);

auto result = index->KnnSearch(
    query, 10,
    R"({"sindi_v2": {
        "n_candidate": 100,
        "query_prune_ratio": 0.1,
        "term_prune_ratio": 0.2,
        "term_retain_threshold": 10000,
        "use_term_lists_heap_insert": true
    }})").value();
```

## 构建参数

构建参数放在 `index_param` 下。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | int | 必填 | 单条稀疏向量允许的最大非零项数量，不是词表大小。 |
| `term_id_limit` | int | `1000000` | term ID 上界，最高为 50 000 000。 |
| `window_size` | int | `50000` | 每个 window 的文档数，范围为 10 000 到 60 000。 |
| `doc_prune_ratio` | float | `0.0` | 构建时丢弃最低权重文档 term 的比例，范围为 `[0.0, 1.0)`。 |
| `use_quantization` | bool 或 string | `false` | `false` 存 FP32，`true` 存 SQ8，`"fp16"` 存 FP16。 |
| `use_reorder` | bool | `false` | 保存高精度向量并对粗排候选重排。 |
| `rerank_type` | string | `"fp32"` | 重排存储类型：`fp32` 或 `dmq8`。 |
| `dmq_shared_codebook_threshold` | int | `1024` | 低频 term 使用共享 DMQ codebook 的阈值。 |
| `remap_term_ids` | bool | `false` | 压缩稀疏或间隔很大的外部 term ID。 |
| `avg_doc_term_length` | int | `100` | 仅用于内存估算。 |
| `immutable` | bool | `false` | 选择紧凑的只读内存 term DataCell。 |
| `term_io` | object | `{"type": "reader_io"}` | 加载后保存 term-first posting 的后端。 |
| `rerank_io` | object | `{"type": "block_memory_io"}` | 保存重排向量的后端。 |
| `rerank_layout` | uint32 | `0` | top-terms-signature 布局使用的前导 term 数量；`0` 表示关闭。 |

文件型 `term_io` 支持 `mmap_io`、`buffer_io` 和 `async_io`，并要求设置
`file_path`。文件型 `rerank_io` 未设置 `file_path` 时，VSAG 使用
`<term_io.file_path>.rerank`。

`rerank_layout > 0` 要求 `use_reorder: true`。`rerank_type: "dmq8"` 要求
`rerank_layout: 0`，并使用默认的 `block_memory_io` 重排后端。

## 检索参数

检索参数放在 `{"sindi_v2": {...}}` 下。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_candidate` | int | `0` | 粗排候选数量；`0` 使用根据 `topk` 推导的索引默认值。 |
| `query_prune_ratio` | float | `0.0` | 跳过最低权重查询 term 的比例，范围为 `[0.0, 1.0)`。 |
| `term_prune_ratio` | float | `0.0` | 每条 term list 中跳过最低存储权重的比例，范围为 `[0.0, 1.0)`。 |
| `term_retain_threshold` | uint64 | `0` | 单个 term 在所有 window 中最多扫描的 posting 总数；`0` 表示关闭，正数使每个非空 window posting list 最多扫描 `max(1, floor(threshold / window_count))` 条。 |
| `use_term_lists_heap_insert` | bool | `true` | 是否选择基于 term list 的候选插入路径。 |

合并 ratio 与 threshold 限制后，每条非空 term list 至少扫描一个 posting。

## 如何选择 SINDI 与 SINDI_V2

完整稀疏索引常驻内存、且 window-first 序列化合适时选择 SINDI。需要按 term
组织持久化 posting，或希望搜索时只从外部存储读取查询 term list 时选择 SINDI_V2。
