# 搜索请求与过滤器

本页介绍描述*如何*搜索的类型：统一的 [`SearchRequest`](#searchrequest)、过滤原语
[`Filter`](#filter) 与 [`Bitset`](#bitset)，以及用于增量搜索的
[`IteratorContext`](#iteratorcontext)。已废弃的 [`SearchParam`](#searchparam已废弃) 在末尾给出以便
迁移。

## `SearchRequest`

声明于 `vsag/search_request.h`。`SearchRequest` 是一个普通结构体，打包了
[`Index::SearchWithRequest`](index_class.md#searchwithrequest) 的每一个选项。填入你需要的字段，其余
保持默认即可。

```cpp
vsag::SearchRequest request;
request.query_ = query;      // 单个查询，或 HGraph/IVF 的连续 KNN 查询批次
request.mode_ = vsag::SearchMode::KNN_SEARCH;
request.topk_ = 10;
request.params_str_ = R"({"hgraph": {"ef_search": 100}})";

auto result = index->SearchWithRequest(request);
```

### `SearchMode`

```cpp
enum class SearchMode {
    KNN_SEARCH = 1,    // 返回 top-k 个最近向量
    RANGE_SEARCH = 2,  // 返回 radius_ 范围内的所有向量
};
```

### 基础字段

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `query_` | `DatasetPtr` | `nullptr` | 查询。HGraph 和 IVF 的 KNN 支持连续的多查询批次；范围搜索只支持单个查询。 |
| `mode_` | `SearchMode` | `KNN_SEARCH` | KNN 还是范围搜索。 |
| `topk_` | `int64_t` | `10` | 要返回的邻居数（KNN 模式）。必须为正。 |
| `radius_` | `float` | `0.5` | 距离阈值（范围模式）。非负。 |
| `limited_size_` | `int64_t` | `-1` | 范围结果的上限；`-1` 表示不限。 |
| `params_str_` | `std::string` | `""` | 算法特有的搜索参数 JSON（如 `ef_search`）。 |

### 自定义查询距离回调

`distance_batch_func_` 可选地让应用提供 query 到向量的分数。它接收稳定的外部向量 ID，
因此闭包可以在 VSAG 之外持有 query 和该 query 的上下文：

```cpp
request.distance_batch_size_ = 32;
request.distance_batch_func_ = [query](const int64_t* ids, uint64_t count, float* scores) {
    for (uint64_t i = 0; i < count; ++i) {
        scores[i] = Score(query, ids[i]);
    }
};
```

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `distance_batch_func_` | `SearchDistanceBatchFunc` | `nullptr` | 按输入外部 ID 的顺序填写对应分数。分数越小越好，且必须是有限值。 |
| `distance_batch_size_` | `uint64_t` | `1` | 单次回调的最大 ID 数。批量推理或外部数据访问应设置为 scorer 的高效批大小；设置回调时必须为正。 |

回调仅属于当前请求，不参与序列化。它可以捕获 query。`hgraph` 的回调模式允许 `query_` 为 null，
但 `ivf` 仍需要一个查询向量用于桶路由。回调在一次请求内必须稳定；若调用方启用并行执行则必须
线程安全；不得抛异常，且只能写入有限分数。

`brute_force` 用该回调执行精确 KNN 和范围搜索。`hgraph` 仅支持 KNN：回调驱动图遍历和最终
排序，但图仍由配置的内置 metric 构建，因此无法保证该回调分数下的 recall。HGraph 可能为保持
图连通性而计算被过滤的遍历节点，但不会返回这些节点。
回调 HGraph 不支持并行检索或 `brute_force_threshold`；这些配置会被拒绝，而不会静默改变检索语义。

`ivf` 支持带回调的 KNN，但必须提供一个非空的单查询向量。IVF 先以配置的内置 metric 选出
`scan_buckets_count` 个桶，再对这些桶中的候选调用回调。回调决定候选排序，但无法召回未被选中的
桶内向量；增大 `scan_buckets_count` 可提高 recall。回调 IVF 不支持范围搜索、`disable_bucket_scan`、
bucket graph 或并行搜索，这些配置会被拒绝。回调模式会自动关闭精排。其他索引不支持该回调。

### IVF 桶路由

IVF 可通过 `params_str_` 接收
`{"ivf":{"scan_buckets_count":N,"disable_bucket_scan":true}}`。该仅路由模式按查询返回
`N` 个 bucket ID，而非向量 label。`NumElements()` 为查询数，`Dim()` 为
`scan_buckets_count`，`GetIds()` 包含桶 ID（空槽位为 `-1`），
`GetDistances()` 为到各桶中心的距离。不扫描桶内向量，因此过滤器、`topk`、范围限制、精排和
reasoning 选项均会被忽略。

### IVF 桶 ID 旁路

`bucket_ids_` 字段允许调用方提供预选的桶 ID，绕过 IVF 的内部桶路由（`ClassifyDatasForSearch`）。
当非空时，搜索跳过基于质心的选择，仅扫描指定的桶。

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `bucket_ids_` | `std::vector<std::vector<int64_t>>` | `{}` | 用于绕过 IVF 路由的预选桶 ID。 |

**语义：**
- **空**（默认）：使用正常的桶路由（`ClassifyDatasForSearch`）。
- **非空**：跳过路由，仅扫描提供的桶 ID。

**约束：**
- IVF 批量 KNN 要求外层向量与查询一一对应，每个查询恰好一个条目；其他受支持的场景仍为
  单查询，并要求外层向量恰好包含一个条目。
- 每个桶 ID 必须在 `[0, bucket_count)` 范围内。
- 重复的桶 ID 会被拒绝。
- 与 `disable_bucket_scan` 模式不兼容。
- 调用方负责桶 ID 的排序（不会自动排序）。

### 过滤字段

有三种过滤机制可用；当启用了多于一种时，它们以逻辑**与（AND）**组合。

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `enable_attribute_filter_` | `bool` | `false` | 启用 SQL 风格的属性过滤。 |
| `attribute_filter_str_` | `std::string` | `""` | 过滤表达式（见下）。需要 `enable_attribute_filter_`。 |
| `enable_filter_` | `bool` | `false` | 启用自定义 [`Filter`](#filter) 回调。 |
| `filter_` | `FilterPtr` | `nullptr` | filter 对象。需要 `enable_filter_`。 |
| `enable_bitset_filter_` | `bool` | `false` | 启用 [`Bitset`](#bitset) 过滤。 |
| `bitset_filter_` | `BitsetPtr` | `nullptr` | bitset。`Test(id) == true` 表示**排除**该 id。需要 `enable_bitset_filter_`。 |

`attribute_filter_str_` 的语法类似 SQL。示例：

```text
category = 'electronics' AND price != 1000
multi_in(category, ['electronics', 'clothing']) AND multi_notin(color, ['red', 'blue'])
```

见 [属性过滤（混合搜索）](../advanced/attribute_filter.md) 与
[带过滤的搜索](../advanced/filtered_search.md)。

### 资源与迭代器字段

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `search_allocator_` | `Allocator*` | `nullptr` | 逐次搜索 allocator；为 null 时回退到索引 allocator。 |
| `enable_iterator_search_` | `bool` | `false` | 启用增量（迭代式）搜索。 |
| `p_iter_ctx_` | `IteratorContext**` | `nullptr` | 迭代状态的句柄，跨调用复用。 |
| `is_last_search_` | `bool` | `false` | 标记迭代序列的最后一次调用。 |
| `expected_labels_` | `std::vector<int64_t>` | `{}` | 期望出现在结果中的 id；启用对漏召回的推理分析。 |

见 [搜索路径 Allocator](../advanced/search_allocator.md) 与
[迭代式搜索](../advanced/iterator_search.md)，allocator 示例见 `examples/cpp/313`/`314`。

## `Filter`

声明于 `vsag/filter.h`。实现这个抽象类以表达任意的“是否保留该 id？”逻辑。通过 `FilterPtr`
（`std::shared_ptr<Filter>`）持有它。

```cpp
class Filter {
public:
    enum class Distribution { NONE = 0, RELATED_TO_VECTOR };

    virtual bool CheckValid(int64_t id) const = 0;          // true  => 保留该 id
    virtual bool CheckValid(const char* data) const;         // extra-info 变体（默认 true）
    virtual float ValidRatio() const;                        // 保留比例（默认 1.0）
    virtual Distribution FilterDistribution() const;         // 提示（默认 NONE）
    virtual void GetValidIds(const int64_t** valid_ids, int64_t& count) const;
};
```

> **约定：** `Filter::CheckValid(id)` 返回 `true` 表示**保留**该向量。这与
> [`Index`](index_class.md#knnsearch-重载) 上的 `bitset` / `std::function<bool(int64_t)>` 预过滤重载
> 相反 —— 在那些重载里 `true` 表示*被过滤掉*。选择重载时请牢记这一区别。

| 成员 | 用途 |
|------|------|
| `CheckValid(int64_t id)` | 核心谓词。`true` 使该 id 保留在结果中。 |
| `CheckValid(const char* data)` | 对元素 extra-info 字节的谓词。默认为 `true`。 |
| `ValidRatio()` | 预估通过的向量比例；让引擎选择策略。 |
| `FilterDistribution()` | `RELATED_TO_VECTOR` 提示有效性与向量位置相关。 |
| `GetValidIds(...)` | 可选地暴露显式的有效 id 集合。 |

见 `examples/cpp/301_feature_filter.cpp`。

## `Bitset`

声明于 `vsag/bitset.h`。一个按位置索引的紧凑位标志集合，通过 `BitsetPtr` 持有。它既用作过滤输入，也可
作为工具（如 [`l2_and_filtering`](types.md#工具函数) 的返回值）。

```cpp
static BitsetPtr Random(int64_t length);  // 给定长度的随机 bitset
static BitsetPtr Make();                  // 空 bitset

void Set(int64_t pos, bool value);
void Set(int64_t pos);       // = Set(pos, true)
bool Test(int64_t pos) const;
uint64_t Count();            // 置位的数量
std::string Dump();          // 调试转储
```

> 当 `Bitset` 被用作搜索预过滤（`bitset_filter_`，或 `KnnSearch` / `RangeSearch` 的 `invalid` 参数）
> 时，`Test(id) == true` 表示该 id 被**过滤掉**。

## `IteratorContext`

声明于 `vsag/iterator_context.h`。一个不透明句柄，保存进行中的迭代式搜索的位置，使后续调用能从上一次
停止处继续。

```cpp
class IteratorContext {
public:
    virtual ~IteratorContext() = default;
};
```

你无需直接构造或检查它。VSAG 在首次迭代式搜索时分配它；在之后每次调用中把同一个句柄传回
（通过 `SearchRequest::p_iter_ctx_`，或 `KnnSearch` 的迭代重载），并在最后一次调用时设置
last-search 标志，以便引擎释放它。见 [迭代式搜索](../advanced/iterator_search.md)。

## `SearchParam`（已废弃）

声明于 `vsag/search_param.h`。`SearchParam` 早于 `SearchRequest`，仅为已废弃的
`KnnSearch(query, k, SearchParam&)` 重载而保留。

```cpp
struct SearchParam {  // [[deprecated]] 请改用 SearchRequest
    bool is_iter_filter{false};
    bool is_last_search{false};
    const std::string& parameters;
    FilterPtr filter{nullptr};
    Allocator* allocator{nullptr};
    IteratorContext* iter_ctx{nullptr};
};
```

**所有新代码请优先使用 [`SearchRequest`](#searchrequest) +
[`SearchWithRequest`](index_class.md#searchwithrequest)。** `SearchParam` 以引用方式持有
`parameters`，因此被引用的字符串必须比该调用活得更久。

## 参见

- [Index](index_class.md) —— 消费这些类型的搜索方法。
- [Dataset](dataset.md) —— 构造 `query_` 并读取结果。
- [辅助类型](types.md) —— 属性过滤所用的属性值类型。
