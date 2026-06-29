# 索引构建与训练

VSAG 把索引构建拆成三个阶段：

1. **Train** —— 在样本数据上拟合内部量化器 / 分区器。
2. **Add** —— 用训练好的编码器把向量插入索引。
3. **Build** —— 一站式包装：在同一份数据上先 `Train` 再 `Add`。

绝大多数用户只需要调用 `Build`。下面两种情况值得单独说明：

- **`Train` + 增量 `Add`。** 当语料规模大或者数据是分批到达时，可以先用代表性样本训练，再通过
  `Add` 流式追加（无需重建索引）。参考
  [`examples/cpp/311_feature_train.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/311_feature_train.cpp)。
- **ODescent。** HGraph / Pyramid 的另一种构图算法，采用批量迭代精修而非逐条插入。参考
  [`examples/cpp/312_feature_odescent.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/312_feature_odescent.cpp)。

## `Train` API

```cpp
tl::expected<void, Error> Index::Train(const DatasetPtr& data);
```

声明位置 `include/vsag/index.h`。在（通常是抽样的）数据集上训练索引，但**不**写入这些
向量。返回 `tl::expected<void, Error>`，使用 `.has_value()` 判断成功与否。

具备实质训练逻辑的索引：**HGraph**、**IVF**、**BruteForce**、**WARP**、**Pyramid**。对它们
来说，`Build(data)` 会先训练再写入向量 —— 默认 NSW 构图模式下相当于 `Train(data)` 之后再
`Add(data)`，而当 HGraph / Pyramid 配置 `graph_type: "odescent"` 时，写入阶段会走 ODescent
的批量构图路径，而不是逐条 `Add`（见 `src/algorithm/` 下的 `HGraph::build_by_odescent` /
`Pyramid::Build`）。

### 何时需要单独调用 `Train`

- 基础量化器需要训练。能力标志
  [`IndexFeature::NEED_TRAIN`](https://github.com/antgroup/vsag/blob/main/include/vsag/index_features.h)
  在 HGraph 与 IVF 中可靠反映这一点：HGraph 当 `base_quantization_type` **不是** `fp32` /
  `fp16` / `bf16` 时设置（`src/algorithm/hgraph.cpp:1803`）；IVF 始终设置
  （`src/algorithm/ivf.cpp:316`），因为其聚类中心必须训练。Pyramid 目前在 `InitFeatures()`
  中**不会**设置 `NEED_TRAIN`，即使其内部 HGraph 量化器需要训练，因此请勿依赖
  `HasFeature(NEED_TRAIN)` 来判断 Pyramid —— 当你选用需要训练的 `base_quantization_type`
  时请显式调用 `Train`。fp32 / fp16 / bf16 不需要训练（即使调用了 `Train` 也是无副作用的
  空操作）。
- 希望分多批次写入向量，而不是一次性通过 `Build` 写完。
- 希望导出已训练的模型供其他索引实例复用（通过 `ExportModel`）。

### 用法：训练一次，流式追加

```cpp
auto params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "max_degree": 32,
        "ef_construction": 100,
        "base_quantization_type": "sq8"
    }
})";
auto index_result = vsag::Factory::CreateIndex("hgraph", params);
if (!index_result.has_value()) {
    std::cerr << "Create index failed: " << index_result.error().message << std::endl;
    return -1;
}
auto index = index_result.value();

// 第 1 步 —— 在全量（或代表性样本）上训练。
auto train_result = index->Train(base);
if (!train_result.has_value()) {
    std::cerr << "Train failed: " << train_result.error().message << std::endl;
    return -1;
}

// 第 2 步 —— 逐条或小批量追加向量。
for (int64_t i = 0; i < num_vectors; ++i) {
    auto one = vsag::Dataset::Make();
    one->NumElements(1)
       ->Dim(dim)
       ->Ids(ids + i)
       ->Float32Vectors(vectors + i * dim)
       ->Owner(false);
    auto add_result = index->Add(one);
    if (!add_result.has_value()) { /* handle */ }
}
```

完整示例见
[`examples/cpp/311_feature_train.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/311_feature_train.cpp)。

### `Train` / `Build` / `Add` 三者对比

| 调用 | 是否训练量化器？ | 是否写入向量？ | 适用场景 |
|------|-----------------|---------------|----------|
| `Build(data)` | 是 | 是（写入全部 `data`） | 一次性批量加载：手头已经有完整数据集。 |
| `Train(data)` | 是 | 否 | 之后需要分批写入向量。 |
| `Add(data)` | 否（需先 `Train` 或 `Build`） | 是 | 索引已训练后的增量写入。 |

## ODescent：另一种构图算法

HGraph 与 Pyramid 默认使用 **NSW 风格** 构图 —— 每条向量逐条插入，在插入时通过搜索找到邻居
并建边（`graph_type: "nsw"`）。**ODescent**（"Optimized NN-Descent"）是另一种实现：先在
完整数据集上初始化一张随机 k-NN 图，然后通过若干轮采样候选交换迭代精修边。

在大批量构建场景下，ODescent 通常能在召回率相当的情况下显著降低构图开销，因为精修循环可以
在数据维度上整齐并行，避免了逐条插入时的单点搜索。

ODescent 的实现位于 `src/impl/odescent/odescent_graph_builder.{h,cpp}`，目前被
**HGraph**、**Pyramid**（构图路径）使用。

### 在 HGraph 中启用 ODescent

在 HGraph 的 `index_param` 中加入 `graph_type: "odescent"`：

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26,
        "ef_construction": 100,
        "graph_type": "odescent",
        "graph_iter_turn": 10,
        "neighbor_sample_rate": 0.3,
        "alpha": 1.2
    }
}
```

然后正常调用 `Build(data)` 即可，无需其他 API 调整。完整示例见
[`examples/cpp/312_feature_odescent.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/312_feature_odescent.cpp)。

### ODescent 构图参数

下列键放在 `index_param` 中，与常规 HGraph 参数并列：

| 参数 | 默认值（HGraph 模板） | 说明 |
|------|---------------------|------|
| `graph_type` | `"nsw"` | 设为 `"odescent"` 启用该构图算法。 |
| `graph_iter_turn` | `30` | 精修迭代轮数。值越大图质量越高，但构图越慢。 |
| `neighbor_sample_rate` | `0.2` | 每轮迭代中从每个节点邻居采样的比例（用于候选交换）。 |
| `alpha` | `1.2` | 多样性剪枝阶段的 α 因子。值越大边越稀疏、多样性越强。 |
| `min_in_degree` | `1` | 剪枝后修复阶段所保证的最小入度。 |
| `build_block_size` | `10000` | 并行粒度（每个 worker 处理的向量数）。 |

`max_degree` 沿用 HGraph 顶层配置，无需在 ODescent 这里重复指定；图的上层会自动使用
`max_degree / 2`。

### ODescent vs NSW 如何选择

- **选 ODescent**：已经有完整数据集，并希望充分利用多核机器加速构图。批量精修比逐条插入的
  并行度更高。
- **选 NSW**（默认）：需要增量构建索引，或希望构图阶段内存占用尽量小，又或者尚未观察到构图
  耗时的瓶颈。

两种算法构出的图在查询期完全等价，所有搜索参数（`ef_search`、`pq_rerank` 等）保持不变。

## 参考

- [创建索引](../guide/create_index.md)
- [HGraph 索引参数](../indexes/hgraph.md)
- [Pyramid 索引参数](../indexes/pyramid.md)
- [索引参数参考](../resources/index_parameters.md)
