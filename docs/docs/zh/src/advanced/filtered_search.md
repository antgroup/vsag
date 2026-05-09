# 带过滤的搜索

带过滤的搜索（Filtered Search）允许在 `KnnSearch` 或 `RangeSearch` 中只保留满足应用自定义条件
的向量。当底层索引算法支持时，VSAG 会在图遍历**过程中**应用该谓词，从而避免“先取 top-k 再丢弃”
所带来的召回率损失与额外延迟。

本文介绍三种基于 id 的过滤 API：

- **位图过滤（Bitset filter）**：以向量 id 作为下标的紧凑位数组。
- **函数回调过滤（Function callback）**：`std::function<bool(int64_t)>`。
- **`Filter` 对象**：继承自 `vsag::Filter` 的子类，除了判定逻辑之外还可以向算法暴露
  有效占比、分布等提示信息。

如果谓词是结构化字段上的 SQL 风格表达式，请阅读
[属性过滤（混合搜索）](attribute_filter.md)；如果是基于每条向量的不透明字节负载在图内过滤，
请阅读 [Extra Info](extra_info.md)。

> 注意：本文与 [内存-磁盘混合索引](hybrid_index.md) 无关，后者描述的是 DiskANN 的存储布局，
> 而非搜索阶段的过滤。

## 真值约定

三种 API 关于「这个 id 是否被排除」的语义并不一致，混用前请仔细对照下表。

| API                  | 方法                | 返回 `true` 表示…   |
|----------------------|---------------------|---------------------|
| `Bitset`             | `Test(id)`          | 该 id 被**过滤掉** |
| `std::function`      | `f(id)`             | 该 id 被**过滤掉** |
| `Filter::CheckValid` | `CheckValid(id)`    | **保留**该 id      |

位图与 `std::function` 两种重载在内部都会被包装为 `BlackListFilter`
（见 `src/impl/filter/black_list_filter.cpp`）：位被置上、或回调返回 `true`，都表示该 id
被排除。`Filter::CheckValid` 则相反——返回 `true` 表示**保留**。如果你已经维护了一份
「删除 id 位图」，最自然的方式是位图过滤；如果是任意谓词逻辑、并且能提供有效占比等提示，
`Filter` 对象会更合适。

## 位图过滤

`vsag::Bitset`（`include/vsag/bitset.h`）是按序号下标的可增长位数组。

```cpp
auto invalid = vsag::Bitset::Make();
for (int64_t i = 0; i < num_vectors; ++i) {
    if (ids[i] % 2 == 0) {
        invalid->Set(ids[i]);    // 偶数 id 被排除
    }
}

auto search_params = R"({ "hnsw": { "ef_search": 100 } })";
auto result = index->KnnSearch(query, /*topk=*/10, search_params, invalid).value();
```

位图按向量 id 索引，但查询时 id 会被掩码到低 32 位
（`bit_index = id & ROW_ID_MASK`，`ROW_ID_MASK = 0xFFFFFFFFLL`，见
`src/impl/filter/black_list_filter.cpp`）。低 32 位相同的两个 id 会在位图中冲突，因此使用
位图过滤时请把 id 控制在 `[0, 2^32)`，否则改用 `Filter` 对象。位图按 id 索引而非按插入
顺序；如果应用层会复用 id，请自行处理一致性。

## 函数回调过滤

直接使用 lambda 或 `std::function<bool(int64_t)>` 即可。回调返回 `true` 表示该 id 被
**排除**（内部会被包装成 `BlackListFilter`）：

```cpp
// 排除偶数 id：返回 true 即被过滤掉。
std::function<bool(int64_t)> drop_even = [](int64_t id) { return id % 2 == 0; };
auto result = index->KnnSearch(query, 10, search_params, drop_even).value();
```

适合写少量自定义逻辑而不需要继承类的场景。如果你更习惯「返回 true 表示保留」的写法，
请改用 `Filter` 对象。

## `Filter` 对象

最完整的 API 是 `vsag::Filter`（`include/vsag/filter.h`）。当算法可以利用谓词的额外提示
（如有效占比）时，建议继承它：

```cpp
class MyFilter : public vsag::Filter {
public:
    bool CheckValid(int64_t id) const override {
        return id % 2 == 1;
    }

    // 谓词通过率的近似估计；搜索算法据此调整候选缓冲区大小，
    // 估计准确可同时改善延迟与召回率。
    float ValidRatio() const override { return 0.5F; }

    // 通过的 id 是否在向量空间中聚集。
    // NONE 表示「无关」；如果谓词与向量位置相关（例如地理标签），用 RELATED_TO_VECTOR。
    Distribution FilterDistribution() const override { return Distribution::NONE; }
};

auto filter = std::make_shared<MyFilter>();
auto result = index->KnnSearch(query, 10, search_params, filter).value();
```

主要方法：

| 方法 | 默认实现 | 用途 |
|---|---|---|
| `CheckValid(int64_t id)` | 纯虚 | 必填。返回 `true` 表示保留该 id。 |
| `CheckValid(const char* data)` | 返回 `true` | 用于在图内基于 extra_info 字节负载过滤，参见 [Extra Info](extra_info.md)。 |
| `ValidRatio()` | `1.0F` | `[0, 1]` 区间内的有效占比提示。 |
| `FilterDistribution()` | `NONE` | `NONE` 或 `RELATED_TO_VECTOR`。 |
| `GetValidIds(...)` | 空实现 | 极端选择性谓词下的可选白名单接口。 |

`ValidRatio` 估计错误不会导致结果错误，但偏大会增大延迟、偏小会拉低召回率。

## 重载列表

`KnnSearch` 与 `RangeSearch` 都提供四种过滤形态（`include/vsag/index.h`）：

```cpp
// KnnSearch
index->KnnSearch(query, topk, params);                                    // 不过滤
index->KnnSearch(query, topk, params, BitsetPtr invalid);
index->KnnSearch(query, topk, params, std::function<bool(int64_t)> f);
index->KnnSearch(query, topk, params, FilterPtr filter);

// RangeSearch
index->RangeSearch(query, radius, params, limited_size);                  // 不过滤
index->RangeSearch(query, radius, params, BitsetPtr invalid, limited_size);
index->RangeSearch(query, radius, params, std::function<bool(int64_t)> f, limited_size);
index->RangeSearch(query, radius, params, FilterPtr filter, limited_size);
```

`limited_size` 是 `RangeSearch` 返回结果的最大数量：

- `limited_size < 0`：不限制（默认 `-1`）。
- `limited_size == 0`：API 会显式拒绝（`CHECK_ARGUMENT(limited_size != 0, ...)`），
  「不限制」请传 `-1`。
- `limited_size > 0`：限定结果列表最多这么多条。

也支持迭代式过滤搜索：

```cpp
vsag::IteratorContext* ctx = nullptr;
index->KnnSearch(query, topk, params, filter, ctx, /*is_last_search=*/false);
// 用同一个 ctx 反复调用；最后一次调用时把 is_last_search 置为 true 以释放上下文。
```

## 索引支持矩阵

所有索引类型都接受位图、函数与 `FilterPtr` 三种形式——内部会把位图与 lambda 自动包装成
`FilterPtr`。下表中的列对应每个索引登记的**能力标志**（见
`include/vsag/index_features.h`），运行时 `CheckFeature` 返回的也是这些。

| 索引         | `_KNN_SEARCH_WITH_ID_FILTER` | `_RANGE_SEARCH_WITH_ID_FILTER` | `_KNN_ITERATOR_FILTER_SEARCH` |
|--------------|:----------------------------:|:------------------------------:|:-----------------------------:|
| HGraph       |             支持             |              支持              |              支持             |
| HNSW         |             支持             |              支持              |              支持             |
| IVF          |             支持             |              支持              |               —               |
| BruteForce   |             支持             |              支持              |               —               |
| DiskANN      |             支持             |              支持              |               —               |
| Pyramid      |             支持             |              支持              |               —               |
| SINDI / WARP |             支持             |              支持              |               —               |

基于 id 的过滤可在运行时通过
`index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)`、
`SUPPORT_RANGE_SEARCH_WITH_ID_FILTER`、`SUPPORT_KNN_ITERATOR_FILTER_SEARCH` 查询。
`SUPPORT_KNN_SEARCH_WITH_EX_FILTER` 与本文无关，它对应的是基于 extra_info 字节负载的
过滤，详见 [Extra Info](extra_info.md)。

## 性能要点

- 谓词越严格（`ValidRatio` 越小），搜索需要扩展的候选越多。对图索引而言，谓词非常严格时
  应同步增大 `ef_search`，否则当通过率低于约 1% 时召回率会显著下降。
- 位图过滤最快，因为 `Test()` 只是一次位查询。`Filter` 对象内若有重逻辑，需注意它会被
  调用很多次。
- `RangeSearch` 在过滤通过率较高、范围较宽时建议设定一个合理的 `limited_size`，避免结果
  集无界增长。
- 与 [属性过滤](attribute_filter.md) 组合时，使用 `SearchRequest` 即可，所有启用的过滤项
  会按逻辑 AND 连接。

## 通过 `SearchRequest` 组合过滤

`SearchRequest`（`include/vsag/search_request.h`）是 `SearchWithRequest` 的统一入口，
可同时携带位图、`Filter` 对象与属性表达式，所有启用的过滤项按 AND 连接：

```cpp
vsag::SearchRequest req;
req.query_                = query;
req.mode_                 = vsag::SearchMode::KNN_SEARCH;
req.topk_                 = 10;
req.params_str_           = R"({ "hgraph": { "ef_search": 200 } })";
req.enable_filter_        = true;
req.filter_               = std::make_shared<MyFilter>();
req.enable_bitset_filter_ = true;
req.bitset_filter_        = invalid;
auto result = index->SearchWithRequest(req).value();
```

`attribute_filter_str_` 字段的语法见 [属性过滤](attribute_filter.md)。

## 示例

- C++：[`examples/cpp/301_feature_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/301_feature_filter.cpp)
  ——同时演示三种过滤方式（HNSW 上）。
- C++：[`examples/cpp/320_feature_extra_info.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/320_feature_extra_info.cpp)
  ——基于 `CheckValid(const char*)` 字节负载重载的图内过滤。

## Python 状态

过滤 API 暂未暴露到 Python；`examples/python/todo_examples/301_feature_filter.py`
是一个空占位文件。当前请使用 C++ API 进行带过滤的搜索。
