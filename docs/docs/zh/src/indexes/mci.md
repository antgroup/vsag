# MCI

MCI 是 VSAG 中面向稠密向量的索引，它把 K 近邻图和极大团候选结构组合在一起。与纯图游走相比，
MCI 会在构建阶段多做一层候选组织，把邻接关系整理成 clique 风格的候选组，以减少查询阶段真正
需要打分的向量数。

MCI 还提供可选的 **HGraph Hybrid overlay**，用于过滤检索。在这种模式下，序列化落盘的主索引
仍然是 MCI；HGraph 作为一个独立索引通过 `hgraph_index_path` 加载，只在过滤范围足够宽时参与检索。

- 源码：`src/algorithm/mci.{h,cpp}`
- 示例：[`examples/cpp/322_feature_mci_hybrid_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_mci_hybrid_filter.cpp)

## 工作原理

1. **构建或导入 KNN 图。** MCI 从一个候选图开始，图的度数上限由 `mcs` 控制。若 `knng_path`
   为空，MCI 会通过 ODescent 在内部构图；若设置了 `knng_path`，则直接读取外部固定宽度的二进制
   KNNG 文件。
2. **枚举 clique 候选。** 在候选图之上继续整理出极大团风格的候选组，组大小受 `clique_max`
   限制，便于查询时快速跳到紧凑的候选集合。
3. **在候选组内打分。** 查询阶段，MCI 先按 `seed_count` 收集种子点，再扩展 clique 候选，若
   启用了 `use_reorder`，则会对最终候选再做一次精排。
4. **满足条件时切到 HGraph。** 若启用了 `use_hgraph_hybrid`，且过滤器的 `ValidRatio()` 大于等于
   `hgraph_valid_ratio_threshold`，MCI 就可以把这次请求转发给外部 HGraph，而不是继续走自身的
   clique 搜索路径。

## 快速开始

### 构建普通 MCI 索引

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50
    }
})";

auto index = vsag::Factory::CreateIndex("mci", params).value();

// 填充 base 集合（请替换为你自己的数据来源）。
int64_t n = 10000;
std::vector<int64_t> ids(n);
std::vector<float> data(n * 128);
// ... 填充 ids 和 data ...

auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids.data())->Float32Vectors(data.data())->Owner(false);
index->Build(base);

// 构造一个查询向量。
std::vector<float> q(128);
// ... 填充查询向量 q ...

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q.data())->Owner(false);
auto result = index->KnnSearch(
    query, 10, R"({"mci": {"ef_search": 80, "seed_count": 32}})").value();
```

### 启用 HGraph Hybrid overlay

```cpp
std::string hybrid_params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50,
        "use_hgraph_hybrid": true,
        "hgraph_valid_ratio_threshold": 0.2,
        "hgraph_index_path": "/path/to/hgraph.index",
        "hgraph_ef_search": 100,
        "hgraph_index_param": {
            "base_quantization_type": "fp32",
            "graph_type": "odescent",
            "max_degree": 32,
            "alpha": 1.2,
            "graph_iter_turn": 20,
            "neighbor_sample_rate": 0.2
        }
    }
})";

auto hybrid = vsag::Factory::CreateIndex("mci", hybrid_params).value();
std::ifstream input("/path/to/mci.index", std::ios::binary);
hybrid->Deserialize(input);
```

Hybrid **不是** 一个独立的落盘索引类型。真正序列化到磁盘的仍是 MCI 本体；
`hgraph_index_path` 指向的是由 overlay 在加载时额外打开的 HGraph 配套索引。

## 构建参数

MCI 的构建参数放在通用的 `index_param` 对象下。

| 参数 | 类型 | 典型值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | `fp32`、`sq8`、`rabitq` | 主存储使用的量化方式 |
| `base_codes_type` | string | `flatten` | flat data cell 使用的底层编码布局 |
| `max_degree` | int | `16`-`48` | clique / 搜索图的最大出度 |
| `mcs` | int | `64`-`256` | 构建或导入 KNN 图时使用的候选预算 |
| `clique_max` | int | `16`-`64` | 单个 clique 候选组的大小上限 |
| `alpha` | float | `1.2` | 当 MCI 自建 KNN 图时，ODescent 使用的扩张系数 |
| `knng_path` | string | 空 | 可选的固定宽度二进制 KNNG 文件；为空时由 MCI 内部构图 |
| `clique_path` | string | 空 | 可选的预计算 clique 索引文件 |
| `use_reorder` | bool | `false` | 是否保留更高精度副本并对最终候选精排 |

### KNNG 文件格式

当设置 `knng_path` 时，MCI 期望的二进制文件满足：

- 无文件头
- 每个 base 向量对应一行固定宽度记录
- 每行按 `uint32_t` / `InnerIdType` 存储邻居 id
- 所有行的度数一致

[`examples/cpp/322_feature_mci_hybrid_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_mci_hybrid_filter.cpp)
展示了如何从 HGraph 检索结果导出这样一份 `.knng` 文件。

## 搜索参数

查询参数放在 `mci` 对象下。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ef_search` | int | MCI 查询阶段保留的候选数 |
| `seed_count` | int | clique 扩展前收集的种子 id 数 |
| `hops_limit` | int | 可选的搜索 hop 数上限 |
| `rabitq_one_bit_search` | bool | 当底层编码支持时启用 RabitQ lower-bound 搜索模式 |

```cpp
auto result = index->KnnSearch(
    query, 10, R"({"mci": {"ef_search": 120, "seed_count": 64}})").value();
```

## HGraph Hybrid overlay

Hybrid overlay 面向的是 **带过滤的 KNN 检索**，而不是普通无过滤检索。

### 路由规则

仅当以下条件同时满足时，MCI 才会把带过滤请求路由到 HGraph：

- `use_hgraph_hybrid` 为 `true`
- HGraph 配套索引已经成功加载，且元素数量与 MCI 一致
- 请求使用的是 `Filter` 对象，而不是单独的 bitset 黑名单过滤
- `filter->ValidRatio()` 大于等于 `hgraph_valid_ratio_threshold`

否则，请求仍走普通 MCI 路径。

### Hybrid 专用参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `use_hgraph_hybrid` | bool | 开启 HGraph 辅助的过滤检索路由 |
| `hgraph_valid_ratio_threshold` | float | 只有当 valid ratio 达到该阈值时才切到 HGraph |
| `hgraph_index_path` | string | 外部 HGraph 序列化文件路径 |
| `hgraph_ef_search` | int | 当请求里没有显式 `hgraph` 搜索参数时使用的默认 `ef_search` |
| `hgraph_index_param` | object | 在加载外部 HGraph 前，用于实例化该索引对象的构建参数 |

查询完成后，结果统计信息会包含 `mci_hybrid_route`、`mci_hybrid_valid_ratio` 和
`mci_hybrid_threshold`，便于确认这次过滤请求是否真的走了 HGraph 路线。

## 何时使用 MCI

- 稠密向量场景，希望使用更紧凑的候选结构，而不是纯图游走。
- 已经离线构建了 KNN 图，希望通过 `knng_path` 复用这份图数据。
- 带过滤检索场景：窄过滤走 MCI，自身 broad filter 可通过 Hybrid overlay 复用现有
  HGraph 索引。

如果你的工作负载大多是不带过滤的图检索，建议与 [HGraph](hgraph.md) 对比评估。如果你的主要需求
是向量 + 结构化属性谓词，而不是基于 id 的 `Filter` 对象，也可以参考
[属性过滤（混合搜索）](../advanced/attribute_filter.md)。
