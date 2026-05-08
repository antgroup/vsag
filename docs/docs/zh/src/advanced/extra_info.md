# Extra Info（附加信息）

`extra_info` 是与每条向量一同存放在索引内部的、定长的不透明字节负载。它允许把少量与向量配对的
非向量元数据（例如时间戳、类目 id、权限标签、应用自定义字段）直接保存在向量旁边，从而：

- 通过向量 id 直接获取元数据，无需额外的 KV 存储。
- 在不重新插入向量的前提下，原地更新某条向量对应的元数据。
- 在图遍历**过程中**就基于元数据过滤候选，而不是事后再过滤搜索结果。

VSAG 把该负载视为原始字节流，其内存布局、序列化与解释完全由用户自行决定。

## 索引支持情况

| 索引       | Build/Add 时存入 | `GetExtraInfoByIds` | `UpdateExtraInfo` | 图内过滤（`use_extra_info_filter`） | 搜索结果中返回 |
|------------|:----------------:|:-------------------:|:-----------------:|:-----------------------------------:|:--------------:|
| **HGraph** |       支持       |         支持        |        支持       |                 支持                |      支持      |
| IVF        |       支持       |          —          |         —         |                  —                  |        —       |
| SINDI      |       支持       |          —          |         —         |                  —                  |        —       |

只有 HGraph 注册了相关的能力标志位；如需完整体验请使用 HGraph。运行时可通过
`index->CheckFeature(...)` 进行检查。

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

### 在搜索结果中获取（HGraph）

当 `extra_info_size > 0` 时，HGraph 会自动在结果 `Dataset` 中填入每个返回 id 对应的字节负载：

```cpp
auto result = index->KnnSearch(query, k, search_params).value();
const char* infos = result->GetExtraInfos();          // 长度 = result->GetDim() * extra_info_size
```

返回的结果 `Dataset` 中只设置了 `ExtraInfos` 缓冲区，**并没有**设置 `ExtraInfoSize`，
因此 `result->GetExtraInfoSize()` 会返回 `0`。请使用建索引时配置的 `extra_info_size`
来计算偏移和缓冲区长度。

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

## 基于 Extra Info 的图内过滤（HGraph）

在过滤命中率较低的场景下，事后过滤会浪费大量计算。HGraph 可以在图遍历过程中，对每个候选向量
直接调用用户定义的过滤器并传入其 extra_info 字节，从而让被过滤掉的候选根本不进入结果集。

1. 重写 `vsag::Filter` 中接收字节缓冲区的版本：

   ```cpp
   class CategoryFilter : public vsag::Filter {
   public:
       CategoryFilter(uint32_t lo, uint32_t hi) : lo_(lo), hi_(hi) {}
       bool CheckValid(int64_t /*id*/) const override { return true; }   // 该路径下不会被调用
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

`use_extra_info_filter` 为 `true` 时，HGraph 会调用 `CheckValid(const char*)` 而不是
`CheckValid(int64_t)`。可使用
`index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)` 进行能力检查。

## 能力标志

| 标志                                          | 含义                                                   |
|-----------------------------------------------|--------------------------------------------------------|
| `vsag::SUPPORT_GET_EXTRA_INFO_BY_ID`          | 支持 `GetExtraInfoByIds`。                             |
| `vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT`  | 支持线程安全的 `UpdateExtraInfo`。                     |
| `vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER`     | 搜索时支持 `use_extra_info_filter`。                   |

## 注意事项与限制

- 负载是不透明的字节流，序列化/反序列化由用户负责，库内部仅按偏移做 `memcpy`。
- `extra_info_size` 在 Build 时即被固定，并写入序列化后的索引。
- 存储开销为 `extra_info_size * num_elements` 字节，会被计入 `EstimateMemory`。
- 请尽量保持负载紧凑——它会驻留内存，并在图内过滤时被反复读取。
- 该特性目前仅提供 C++ 接口，未提供 Python 绑定。

## 示例

完整的可运行示例位于 `examples/cpp/320_feature_extra_info.cpp`，演示了在 HGraph 上启用
`extra_info`、按 id 获取、图内过滤搜索以及原地更新等用法。
