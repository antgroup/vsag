# 图索引增强

图类索引在"困难查询"（与真实近邻连通性较弱）下可能出现召回率下降。
VSAG 通过 **Conjugate Graph**（共轭图）机制对这类查询进行在线/离线修补，在几乎不增加索引体积的
情况下显著改善尾部召回。

## 启用共轭图

构建时开启：

```json
{
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400,
        "use_conjugate_graph": true
    }
}
```

搜索时通过搜索参数 JSON 中的 `use_conjugate_graph_search` 字段控制是否启用
（`KnnSearch` 并不存在额外的布尔参数重载）：

```cpp
std::string search_param_json = R"({
    "hgraph": {
        "ef_search": 100,
        "use_conjugate_graph_search": true
    }
})";
auto result = index->KnnSearch(query, k, search_param_json);
```

## 工作原理

已知困难查询的精确最近邻标签时，可调用
`Feedback(query, k, search_parameters, global_optimum_id)`；省略 `global_optimum_id` 时，
HGraph 会通过精确扫描计算。离线增强可调用 `Pretrain(base_ids, k, search_parameters)`，
它会在选定的 float32 基础向量及其近邻之间生成查询并反馈失败路径。两个方法均返回新插入的
共轭边数量，重复反馈返回 0。

共轭图保存从局部结果标签到已知全局最优标签的映射，并在 HGraph 常规遍历后补充候选。
`use_conjugate_graph` 默认关闭；未开启时调用 `Feedback` 或 `Pretrain` 会返回
`UNSUPPORTED_INDEX_OPERATION`。索引启用共轭图后，搜索增强默认开启，也可按次搜索关闭。

## 示例

`examples/cpp/304_feature_enhance_graph.cpp` 给出了从构建、训练到对比召回率的完整流程。

## 适用场景

- 数据分布存在稀疏簇或离群点；
- 对 P99 召回敏感的在线场景；
- 希望在不重建索引的前提下小幅提升召回。

## 注意事项

- 启用后构建时间会略有增加。
- 共轭图数据会随索引一并序列化。
- `UpdateId` 会同时更新 HGraph 标签表与共轭边。
- `Pretrain` 当前支持 float32 HGraph；`Feedback` 支持 HGraph 的全部数据类型。
- 与 `Tune` 可以叠加使用，分别作用于路由质量与运行期参数。
