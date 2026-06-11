# 索引自省

VSAG 提供三类自省 API，让调用方可以发现某个索引的能力、对已有向量计算距离，以及读出关于已构建
索引的结构化信息，而无需重新执行一次搜索：

- **`CheckFeature(IndexFeature)`** —— 运行时能力探测。
- **`CalDistanceById(...)`** —— 计算 query 到已存入向量 id 的距离。
- **`GetIndexDetailInfos()` / `GetDetailDataByName(...)`** —— 读取索引各项结构化详情数据。

这些 API 均为只读操作，可与搜索并发调用。

## 能力探测 —— `CheckFeature`

当底层索引实现公布了某项能力时，`index->CheckFeature(vsag::SUPPORT_*)` 返回 `true`。当代码路径
持有一个具体类型未知的 `IndexPtr`（例如用户配置注入、多态存储）时，应使用此 API：

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
    uint64_t est = index->EstimateMemory(100'000);
}

if (not index->CheckFeature(vsag::SUPPORT_DELETE_BY_ID)) {
    // 跳过 / 通过另一个索引以 remove + re-add 方式回退。
}
```

能力标志几乎覆盖了库中所有可选接口：build / add / 序列化变体、各种并发组合、度量类型、属性
过滤、extra-info 过滤、`Clone`、`ExportModel`、`Tune` 等。完整枚举见
`include/vsag/index_features.h`。

可运行示例：`examples/cpp/307_feature_check_features.cpp`。

## 到已有 id 的距离 —— `CalDistanceById`

`CalDistanceById` 计算 query 与索引中**已存在**的一个或多个向量之间的距离，**无需**执行一次
搜索。它适用于 re-rank、A/B 评估、ground-truth 校验，或对已知候选集合做成对距离计算。

提供两个重载：

```cpp
// 稠密向量索引（HGraph、BruteForce、IVF）
auto r = index->CalDistanceById(query_ptr, ids, count, /*calculate_precise_distance=*/true);

// 稀疏向量索引（SINDI）—— 用 Dataset 封装查询
auto query_ds = vsag::Dataset::Make();
query_ds->NumElements(1)->SparseVectors(/* ... */);
auto r = index->CalDistanceById(query_ds, ids, count, /*calculate_precise_distance=*/true);
```

结果 `Dataset` 中 `GetDistances()` 持有 `count` 个距离。若某个 id 无效（不在索引中），对应位置
返回 `-1.0F`。

### `calculate_precise_distance`

末尾的 `bool` 参数在精度与延迟之间做取舍：

| 取值              | 行为                                                                      |
|-------------------|---------------------------------------------------------------------------|
| `true`（默认）    | 使用全精度向量表征。在内存-磁盘混合索引上可能引发磁盘 I/O。              |
| `false`           | 使用搜索路径缓存的量化 / 近似表征。更快、无 I/O。                         |

可运行示例：`examples/cpp/306_feature_calculate_distance_by_id.cpp`。

## 详情数据 —— `GetIndexDetailInfos` / `GetDetailDataByName`

`GetIndexDetailInfos()` 返回一组 `IndexDetailInfo` 记录，描述索引可对外暴露的每一项命名结构化
数据。每条记录包含 `name`、`description` 和一个 `type` 枚举，后者用于选择 `DetailData` 上的
合适访问器。

是否支持取决于索引类型 —— 这两个 API 没有专门的 `SUPPORT_*` flag。`Index` 基类默认抛
`std::runtime_error("Index doesn't support ...")`（`GetIndexDetailInfos` 与
`GetDetailDataByName`，见 `include/vsag/index.h:658,674`）；HGraph / IVF / BruteForce /
Pyramid / SINDI / WARP 通过 `InnerIndexInterface` 提供了实现。调用时请始终处理 `tl::expected`
的 error 分支。

```cpp
auto infos = index->GetIndexDetailInfos().value();
for (const auto& info : infos) {
    std::cout << info.name << " : " << info.description << '\n';
}
```

知道哪些项可用后，调用 `GetDetailDataByName(name, info)` 获取对应类型的数据：

```cpp
vsag::IndexDetailInfo info;
auto detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_NAME_NUM_ELEMENTS, info).value();
int64_t n = detail->GetDataScalarInt64();

detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_NAME_LABEL_TABLE, info).value();
auto table = detail->GetData2DArrayInt64();   // [row][col] int64 矩阵

detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_DATA_TYPE, info).value();
std::string dt = detail->GetDataScalarString();
```

### 数据类型

`info.type` 决定 `DetailData` 上哪一个访问器有效：

| `IndexDetailDataType`     | 访问器                                |
|---------------------------|---------------------------------------|
| `TYPE_SCALAR_INT64`       | `GetDataScalarInt64()`                |
| `TYPE_SCALAR_DOUBLE`      | `GetDataScalarDouble()`               |
| `TYPE_SCALAR_BOOL`        | `GetDataScalarBool()`                 |
| `TYPE_SCALAR_STRING`      | `GetDataScalarString()`               |
| `TYPE_1DArray_INT64`      | `GetData1DArrayInt64()`               |
| `TYPE_2DArray_INT64`      | `GetData2DArrayInt64()`               |

`include/vsag/index_detail_info.h` 中以常量形式给出的标准详情名：

| 常量                                | 典型类型                | 含义                                          |
|-------------------------------------|-------------------------|-----------------------------------------------|
| `INDEX_DETAIL_NAME_NUM_ELEMENTS`    | `TYPE_SCALAR_INT64`     | 索引当前包含的向量数。                          |
| `INDEX_DETAIL_NAME_LABEL_TABLE`     | `TYPE_2DArray_INT64`    | 逐向量的 label 表（如内部 id ↔ 用户 id 映射）。 |
| `INDEX_DETAIL_DATA_TYPE`            | `TYPE_SCALAR_STRING`    | 底层向量数据类型（如 `"float32"`）。           |

具体索引可能额外暴露其他名称；运行期通过 `GetIndexDetailInfos()` 遍历即可发现。可运行示例：
`examples/cpp/317_feature_get_detail_data.cpp`。

## 注意事项与限制

- `CheckFeature` 是常数时间复杂度。相比对不支持的调用做 `try` / `catch`，应优先使用它。
- `CalDistanceById` 要求底层索引保留足够信息以重新计算距离。对于纯量化索引（不保留原始向量），
  即使传入 `calculate_precise_distance = true`，也可能返回量化距离。
- `GetIndexDetailInfos` 与 `GetDetailDataByName` 是只读快照。返回的数值反映调用瞬间的索引状态，
  并发修改可能使其失效。
