# 索引生命周期管理

索引构建完成后，VSAG 提供一组用于原地修改索引或从已有索引派生新索引的操作。本页文档化完整的
生命周期接口：

- `Remove` —— 按 id 删除向量。
- `UpdateVector` / `UpdateId` —— 修改已有向量或重命名其 id。
- `Clone` —— 对已有索引进行深拷贝。
- `ExportModel` —— 将训练好的模型导出为空索引以复用。

每个操作均为可选项，仅当底层索引通过 `index->CheckFeature(...)` 公布对应能力标志时才可用。

## 能力标志

| 操作              | 能力标志                                  | HGraph | IVF | SINDI |
|-------------------|------------------------------------------|:------:|:---:|:-----:|
| `Remove`          | _（暂无专用标志，参见下文）_              |   是   |  —  |   —   |
| `UpdateVector`    | `SUPPORT_UPDATE_VECTOR_CONCURRENT`       |   是   |  —  |  是   |
| `UpdateId`        | `SUPPORT_UPDATE_ID_CONCURRENT`           |   是   |  —  |  是   |
| `Clone`           | `SUPPORT_CLONE`                          |   是   | 是  |   —   |
| `ExportModel`     | `SUPPORT_EXPORT_MODEL`                   |   是   | 是  |   —   |

对于带能力标志的操作，请在调用前通过 `index->CheckFeature(vsag::SUPPORT_*)` 在运行时进行检查；
不支持的索引会返回 `UNSUPPORTED_INDEX_OPERATION`。`Remove` 目前未提供专用能力标志，是否可用
（以及支持哪种模式）参见下一节。

## 删除向量

`Remove` 按 id 删除向量。HGraph 支持两种删除模式，要求不同：

- `RemoveMode::MARK_REMOVE`（默认）：仅通过 label table 写入墓碑标记，**不依赖** `support_force_remove`
  即可调用。该 id 会在后续搜索中被过滤掉，但底层图节点与向量存储仍然保留。
- `RemoveMode::FORCE_REMOVE`：物理重写图并回收存储槽。该模式仅在索引以
  `index_param` 中 `support_force_remove: true` 构建时可用。该开关会启用 force remove 路径及其额外同步；
  若索引未带 `support_force_remove: true` 构建，调用 `FORCE_REMOVE` 会失败。

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 16,
        "ef_construction": 100,
        "support_force_remove": true
    }
}
```

上述 JSON 仅在打算使用 `FORCE_REMOVE` 时是必需的。若只用 `MARK_REMOVE`，可以省略
`support_force_remove` 字段。

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 16,
        "ef_construction": 100
    }
}
```

```cpp
// 提供单 id 与批量两种重载。
index->Remove(id);
index->Remove(std::vector<int64_t>{id1, id2, id3});
```

### 删除模式

可选的 `RemoveMode` 参数用于选择删除策略：

| 模式                                | 行为                                                                |
|-------------------------------------|---------------------------------------------------------------------|
| `RemoveMode::MARK_REMOVE`（默认）   | 对 id 打墓碑标记；速度快，不收缩、不修图。后续搜索会跳过该 id。不要求 `support_force_remove: true`。 |
| `RemoveMode::FORCE_REMOVE`          | 物理删除向量并修复图结构。开销较大。要求索引以 `support_force_remove: true` 构建。 |

`Remove` 返回成功删除的 id 数量。原本不存在的 id 会被静默跳过，不计入返回值。

可运行示例：`examples/cpp/303_feature_remove.cpp`。

## 更新向量与 id

### `UpdateVector`

`UpdateVector(id, new_base, force_update = false)` 在原地替换已有 id 对应的向量数据。默认的
`force_update = false` 模式会做连通性检查：若新向量距离原向量较远（这会破坏图质量），更新会被
**拒绝**，调用方应当退回到 `Remove` + `Add` 方案。

```cpp
std::vector<float> new_vec(dim);  // 填入替换向量
auto upd = vsag::Dataset::Make();
upd->NumElements(1)->Dim(dim)->Ids(&id)->Float32Vectors(new_vec.data())->Owner(false);

auto status = index->UpdateVector(id, upd, /*force_update=*/false);
if (status.has_value() && *status) {
    // 已原地更新
} else if (status.has_value() && not *status) {
    // 被拒绝：新向量距离原向量太远 —— 退回到 remove + add
    index->Remove(id);
    index->Add(upd);
}
```

将 `force_update` 置为 `true` 会跳过检查并强制更新；请谨慎使用，可能损失召回率。

### `UpdateId`

`UpdateId(old_id, new_id)` 重命名已有 id 而不动底层向量。成功返回 `true`，若 `old_id` 不存在
或 `new_id` 已被占用则返回 `false`。

```cpp
index->UpdateId(123, 456);
```

结合 `UpdateVector`、`Remove`、`Add` 的可运行示例：`examples/cpp/305_feature_update.cpp`。

## 克隆索引

`Clone()` 对整个索引做深拷贝 —— 包括向量、图、量化器状态与元数据 —— 返回一个独立的
`IndexPtr`。该克隆体可独立于源索引进行搜索、修改或序列化。

```cpp
auto cloned = index->Clone().value();

// 克隆完成后，两个索引返回的搜索结果完全一致。
auto r1 = index->KnnSearch(query, k, params).value();
auto r2 = cloned->KnnSearch(query, k, params).value();
```

`Clone` 还可选传入自定义 `Allocator`，使克隆索引使用与源不同的内存区 —— 便于把索引交给拥有
自己内存分配器的线程或组件。分配器细节参见 [内存管理](memory.md)。

可运行示例：`examples/cpp/309_feature_clone.cpp`。

## 导出训练模型

`ExportModel()` 返回一个保留了源索引全部训练状态（量化码本、聚类中心、超参数）但**不含**任何
向量的空索引。这是在多个分片、进程或主机之间共享预训练模型而无需重新训练的标准做法。

```cpp
auto exported = index->ExportModel();
if (not exported.has_value()) {
    // 索引不支持 ExportModel —— 处理错误
    return;
}
auto model = *exported;

// 向空模型注入一批新的（可与源不同的）向量。
for (int64_t i = 0; i < num_vectors; ++i) {
    auto one = vsag::Dataset::Make();
    one->NumElements(1)->Dim(dim)->Ids(ids + i)
       ->Float32Vectors(vectors + i * dim)->Owner(false);
    model->Add(one);
}
```

返回的索引行为上等同于一个通过 `Factory::CreateIndex(...)` 新建并在源数据上完成训练的索引 ——
仅每条向量的存储为空。该模式对训练（中心点 k-means）开销占主导的 IVF 类索引尤其有用。

可运行示例：`examples/cpp/310_feature_export_model.cpp`。

## 注意事项与限制

- 当对应的 `*_CONCURRENT` 能力标志被置位时，HGraph 上的 `Remove`、`UpdateVector`、`UpdateId`
  是并发安全的。该标志组还约束与并发搜索、增加之间的安全组合（如
  `SUPPORT_ADD_SEARCH_DELETE_CONCURRENT`）。
- `MARK_REMOVE` 不会释放内存；如需回收空间请使用 `FORCE_REMOVE` 或定期重建索引。
- `Clone` 的开销与索引规模线性相关。如果只需要磁盘快照，对大索引来说更适合采用「序列化 + 由
  专用 reader 反序列化」的方案。
- `ExportModel` 保留训练状态，但**不**保留任何已插入的向量。导出的模型可以在尚未添加任何向量
  之前自由序列化、分发。
