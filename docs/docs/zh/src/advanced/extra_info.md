# Extra Info（附加信息）

`extra_info` 是与每条向量一同存放在索引内部的、定长的不透明字节负载。它允许把少量与向量配对的
非向量元数据（例如时间戳、类目 id、权限标签、应用自定义字段）直接保存在向量旁边，从而：

- 通过向量 id 直接获取元数据，无需额外的 KV 存储。
- 在不重新插入向量的前提下，原地更新某条向量对应的元数据。
- 在搜索过程中基于元数据过滤候选，而不是事后再过滤搜索结果。

VSAG 把该负载视为原始字节流，其内存布局、序列化与解释完全由用户自行决定。

## 索引支持情况

各索引支持的操作如下：

- **HGraph**：支持 Build/Add 时存入、`GetExtraInfoByIds`、`UpdateExtraInfo`、
  `use_extra_info_filter`，以及在搜索结果中返回 extra info。
- **LazyHGraph**：两个阶段都支持与 HGraph 相同的能力。flat 阶段由 BruteForce 提供能力，
  转换后 graph 阶段由 HGraph 提供能力。
- **BruteForce**：支持 Build/Add 时存入、`GetExtraInfoByIds`、`UpdateExtraInfo`、
  `use_extra_info_filter`，以及在搜索结果中返回 extra info。
- **IVF** 和 **SINDI**：支持 Build/Add 时存入 extra info，但不提供获取、更新、
  extra-info 过滤或在搜索结果中返回 extra info。

当 `extra_info_size > 0` 时，HGraph、LazyHGraph 和 BruteForce 会注册相关能力标志位。
运行时可通过 `index->CheckFeature(...)` 进行检查。

## 启用 Extra Info

在创建索引的参数中，添加顶层整型字段 `extra_info_size`，其值为每条向量预留的字节数。索引一旦
建立，该大小即被固定，并随索引一同序列化。

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "extra_info_size": 12,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26,
        "ef_construction": 100
    }
}
```

LazyHGraph 也使用顶层 `extra_info_size`；LazyHGraph 自身参数仍放在 `lazy_hgraph` 对象中：

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "extra_info_size": 12,
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

未设置 `extra_info_size` 或将其设为 `0` 即表示禁用该特性。

## 在 Build / Add 时提供 Extra Info

通过 `Dataset` 的链式接口绑定字节缓冲区。该缓冲区必须连续，第 `i` 条向量的负载位于
`i * extra_info_size` 字节偏移处。

```cpp
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)
    ->Dim(dim)
    ->Ids(ids.data())
    ->Float32Vectors(vectors.data())
    ->ExtraInfos(extra_infos.data())   // 总长度 num_vectors * extra_info_size 字节
    ->ExtraInfoSize(extra_info_size)   // 必须与索引的 extra_info_size 完全一致
    ->Owner(false);

index->Build(base);   // 或 index->Add(base)
```

`ExtraInfoSize` 必须和索引创建时的 `extra_info_size` 完全相等，否则调用会被拒绝。

## 获取 Extra Info

### 在搜索结果中获取

当 `extra_info_size > 0` 时，支持该能力的索引会在结果 `Dataset` 中填入每个返回 id 对应的
字节负载：

```cpp
auto result = index->KnnSearch(query, k, search_params).value();
const char* infos = result->GetExtraInfos();
auto info_size = result->GetExtraInfoSize();
```

请使用 `info_size` 计算返回缓冲区中的偏移。

### 通过 ID 批量获取（`GetExtraInfoByIds`）

调用方需要预先分配 `count * extra_info_size` 字节的缓冲区：

```cpp
if (index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
    std::vector<char> out(count * extra_info_size);
    index->GetExtraInfoByIds(ids, count, out.data());
}
```

若该能力未开启，调用会返回 `UNSUPPORTED_INDEX_OPERATION`。

## 原地更新 Extra Info

无需触碰向量本身，即可更新单条向量的负载：

```cpp
if (index->CheckFeature(vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT)) {
    auto upd = vsag::Dataset::Make();
    upd->NumElements(1)
       ->Ids(&id)
       ->ExtraInfos(buffer.data())
       ->ExtraInfoSize(extra_info_size)
       ->Owner(false);
    index->UpdateExtraInfo(upd);
}
```

数据集必须只包含一条记录，且大小必须匹配。

## 基于 Extra Info 过滤

在过滤命中率较低的场景下，事后过滤会浪费大量计算。HGraph 与 LazyHGraph 可以在图遍历过程中
对每个候选向量直接调用用户定义的过滤器并传入其 extra_info 字节，从而让被过滤掉的候选不进入
结果集。LazyHGraph 在转换前也支持同样的字节负载过滤，此时 flat 阶段会执行精确扫描。

1. 重写 `vsag::Filter` 中接收字节缓冲区的版本：

   ```cpp
   class CategoryFilter : public vsag::Filter {
   public:
       CategoryFilter(uint32_t lo, uint32_t hi) : lo_(lo), hi_(hi) {}
       bool CheckValid(int64_t /*id*/) const override { return true; }
       bool CheckValid(const char* data) const override {
           uint32_t category_id;
           std::memcpy(&category_id, data, sizeof(category_id));
           return category_id >= lo_ && category_id <= hi_;
       }
       float ValidRatio() const override { return 0.5F; }
   private:
       uint32_t lo_, hi_;
   };
   ```

2. 在搜索参数中的 `hgraph` 块开启 `use_extra_info_filter`，并把过滤器传入 `KnnSearch`：

   ```cpp
   std::string search_params = R"({
       "hgraph": {
           "ef_search": 100,
           "use_extra_info_filter": true
       }
   })";
   auto filter = std::make_shared<CategoryFilter>(3, 7);
   auto result = index->KnnSearch(query, k, search_params, filter).value();
   ```

`use_extra_info_filter` 为 `true` 时，搜索路径会调用 `CheckValid(const char*)` 而不是
`CheckValid(int64_t)`。可使用
`index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)` 进行能力检查。

## LazyHGraph 说明

- 创建 LazyHGraph 索引时必须配置 `extra_info_size`；该字段不放在 `lazy_hgraph` 或 `hgraph`
  对象内部。
- flat 阶段写入的 extra info 会在转换时迁移到内部 HGraph。
- `GetExtraInfoByIds`、`UpdateExtraInfo`、搜索结果返回 extra info，以及
  `use_extra_info_filter` 在转换前后都可用。
- 序列化 LazyHGraph 时会保留当前阶段和已存储的 extra info。

## 能力标志

- `vsag::SUPPORT_GET_EXTRA_INFO_BY_ID`：支持 `GetExtraInfoByIds`。
- `vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT`：支持线程安全的 `UpdateExtraInfo`。
- `vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER`：搜索时支持 `use_extra_info_filter`。

## 注意事项与限制

- 负载是不透明的字节流，序列化/反序列化由用户负责，库内部仅按偏移复制。
- `extra_info_size` 在 Build 时即被固定，并写入序列化后的索引。
- 存储开销为 `extra_info_size * num_elements` 字节；支持该存储统计的索引会将其计入
  `EstimateMemory`。
- 请尽量保持负载紧凑，因为 extra-info 过滤时会读取该负载。
- 该特性目前仅提供 C++ 接口，未提供 Python 绑定。

## 示例

完整的可运行示例位于 `examples/cpp/320_feature_extra_info.cpp`，演示了在 HGraph 上启用
`extra_info`、按 id 获取、extra-info 过滤搜索以及原地更新等用法。
